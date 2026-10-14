//! Effets shader écrits par un agent (`docs/shader-effects.md`) : du WGSL qui définit
//! `fn effect(uv: vec2<f32>) -> vec4<f32>`, appliqué en une passe plein cadre par `compose_frame`,
//! donc à l'identique en preview et à l'export.
//!
//! Ce module porte ce qui ne dépend d'aucun backend : le contrat (le prélude et la struct des
//! paramètres, générée), la validation, la borne des boucles, l'estimation du coût en paliers, la
//! sélection des régions actives et le remplissage des constant buffers. Chaque backend traduit le
//! module par naga et le compile une fois, hors de la boucle de rendu.
//!
//! L'effet est une fonction PURE de l'image composée, du temps, du curseur et de ses paramètres :
//! pas d'état d'une frame à l'autre. C'est ce qui garde la preview (qui saute d'une frame à
//! l'autre au seek) et l'export (qui les rend dans l'ordre) d'accord.

use crate::scene::{SceneShaderEffect, ShaderEffectTarget, ShaderParamDecl, ShaderParamType, ShaderParamValue};
use std::hash::{Hash, Hasher};

/// La version du contrat que ce prélude implémente. Un prélude livré ne change plus : une entrée
/// nouvelle s'ajoute à la fin de `FxContext`, un changement de sens crée la version suivante.
pub const CONTRACT: u32 = 1;

/// Les entrées que l'effet lit, en plus de ses paramètres. Placées APRÈS le code de l'agent (et
/// après `FxParams`) : en WGSL l'ordre des déclarations de module est libre, et les erreurs
/// pointent ainsi les lignes que l'agent a écrites. Voir la section C de la spec.
pub const PRELUDE_V1: &str = r#"
struct FxContext {
    resolution: vec2<f32>,
    time: f32,
    aspect: f32,
    region: vec4<f32>,
    cursor: vec4<f32>,
    clicks: array<vec4<f32>, 4>,
    footage: vec4<f32>,
    camera: vec4<f32>,
};
@group(0) @binding(0) var fx_frame: texture_2d<f32>;
@group(0) @binding(1) var fx_sampler: sampler;
@group(0) @binding(2) var<uniform> ctx: FxContext;

fn frame(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(fx_frame, fx_sampler, uv, 0.0);
}

fn fx_px(n: f32) -> f32 {
    return n / 1080.0;
}

fn fx_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn fx_hash(p: vec2<u32>) -> f32 {
    return f32(fx_pcg(p.x ^ fx_pcg(p.y)) >> 8u) / 16777216.0;
}

struct FxVsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn fx_vs(@builtin(vertex_index) i: u32) -> FxVsOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: FxVsOut;
    o.pos = vec4<f32>(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    o.uv = p;
    return o;
}

@fragment
fn fx_fs(v: FxVsOut) -> @location(0) vec4<f32> {
    return effect(v.uv);
}
"#;

/// Combien de clics récents l'effet voit (`ctx.clicks`), du plus récent au plus ancien.
pub const MAX_CLICKS: usize = 4;
/// Jusqu'où un clic reste visible des effets : au-delà, son entrée est vide (`w` = 0).
pub const CLICK_WINDOW_S: f32 = 2.0;
/// Le plafond de sécurité du travail par pixel, toutes boucles imbriquées comprises. Ce n'est pas
/// un budget : c'est l'ordre de grandeur où un GPU intégré atteint le TDR en 1080p.
pub const MAX_ITERATIONS_PER_PIXEL: u64 = 4096;
/// Taille du constant buffer des paramètres : le plafond technique (32) de `vec4`.
pub const PARAMS_BUFFER_BYTES: usize = 32 * 16;

/// Le constant buffer `ctx`, au layout de `FxContext` (144 octets).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FxContext {
    pub resolution: [f32; 2],
    pub time: f32,
    pub aspect: f32,
    pub region: [f32; 4],
    pub cursor: [f32; 4],
    pub clicks: [[f32; 4]; MAX_CLICKS],
    pub footage: [f32; 4],
    pub camera: [f32; 4],
}

/// Ce que le plan du curseur et la géométrie de la frame donnent aux effets, en uv de sortie.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FxFrameInputs {
    pub cursor: Option<[f32; 2]>,
    /// Les clics récents, du plus récent au plus ancien : point et âge en secondes.
    pub clicks: Vec<([f32; 2], f32)>,
    pub footage: [f32; 4],
    pub camera: Option<[f32; 4]>,
}

/// Le palier de coût d'un effet (section C.7 de la spec) : indicatif, jamais un refus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Light,
    Medium,
    Heavy,
    VeryHeavy,
}

impl Tier {
    /// Les seuils, en lectures de `frame()` par pixel. Sur la RTX 4070 Ti, une lecture coûte
    /// ~4 µs par frame en 1080p : léger ≤ 0,06 ms, moyen ≤ 0,14 ms, lourd ≤ 0,54 ms
    /// (`shader_fx_tier_calibration`, rendering-performance.md).
    pub fn of(frame_reads: u64) -> Tier {
        match frame_reads {
            0..=8 => Tier::Light,
            9..=32 => Tier::Medium,
            33..=128 => Tier::Heavy,
            _ => Tier::VeryHeavy,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Tier::Light => "light",
            Tier::Medium => "medium",
            Tier::Heavy => "heavy",
            Tier::VeryHeavy => "very-heavy",
        }
    }
}

/// Ce que la validation sait d'un effet accepté.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectReport {
    /// Lectures de texture par pixel, boucles multipliées par leur borne (le pire des branches).
    pub frame_reads: u64,
    /// Itérations de boucle par pixel, toutes boucles imbriquées comprises.
    pub iterations: u64,
    pub tier: Tier,
}

impl ShaderParamType {
    fn wgsl(self) -> &'static str {
        match self {
            ShaderParamType::Number | ShaderParamType::Toggle => "f32",
            ShaderParamType::Color => "vec4<f32>",
            ShaderParamType::Point => "vec2<f32>",
        }
    }

    /// (taille, alignement) en octets, règles WGSL de l'espace `uniform`.
    fn size_align(self) -> (usize, usize) {
        match self {
            ShaderParamType::Number | ShaderParamType::Toggle => (4, 4),
            ShaderParamType::Color => (16, 16),
            ShaderParamType::Point => (8, 8),
        }
    }
}

const RESERVED_PARAM_NAMES: &[&str] = &["unused"];

/// La struct `FxParams` et son binding, générés depuis les déclarations. WGSL refuse une struct
/// vide : sans paramètre, un champ `unused` en tient lieu.
fn params_source(params: &[ShaderParamDecl]) -> String {
    let mut out = String::from("struct FxParams {\n");
    if params.is_empty() {
        out.push_str("    unused: f32,\n");
    }
    for p in params {
        out.push_str(&format!("    {}: {},\n", p.name, p.ty.wgsl()));
    }
    out.push_str("};\n@group(0) @binding(3) var<uniform> params: FxParams;\n");
    out
}

/// Le module complet que naga compile : le code de l'agent d'abord, pour que ses lignes gardent
/// leur numéro dans les erreurs.
pub fn module_source(user: &str, params: &[ShaderParamDecl]) -> String {
    format!("{user}\n{}{PRELUDE_V1}", params_source(params))
}

fn check_params(params: &[ShaderParamDecl]) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    for p in params {
        let valid = p.name.len() <= 32
            && p.name.starts_with(|c: char| c.is_ascii_lowercase())
            && p.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        if !valid {
            return Err(format!("parameter `{}`: names are snake_case, a-z first, at most 32 characters", p.name));
        }
        if RESERVED_PARAM_NAMES.contains(&p.name.as_str()) || !seen.insert(p.name.as_str()) {
            return Err(format!("parameter `{}` is declared twice or reserved", p.name));
        }
    }
    Ok(())
}

/// Le module parsé, validé, et sa première fonction d'entrée de fragment.
pub struct Parsed {
    pub module: naga::Module,
    pub info: naga::valid::ModuleInfo,
    pub report: EffectReport,
}

/// Valide un effet comme l'agent doit le lire : paramètres, syntaxe et types (naga), bornes des
/// boucles, puis estimation du coût. L'erreur porte la ligne et la colonne du code de l'agent.
pub fn analyze(user: &str, params: &[ShaderParamDecl]) -> Result<Parsed, String> {
    check_params(params)?;
    let src = module_source(user, params);
    let module = naga::front::wgsl::parse_str(&src).map_err(|e| e.emit_to_string(&src))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|e| e.emit_to_string(&src))?;
    let entry = module
        .entry_points
        .iter()
        .find(|e| e.name == "fx_fs")
        .ok_or("the module has no fx_fs entry point")?;
    let mut costs = CostWalk { module: &module, src: &src, by_function: Default::default() };
    let cost = costs.block(&entry.function, &entry.function.body)?;
    if cost.iterations > MAX_ITERATIONS_PER_PIXEL {
        return Err(format!(
            "the loops run {} iterations per pixel, over the safety ceiling of {MAX_ITERATIONS_PER_PIXEL}: \
             past it a GPU can stall long enough for the system to reset it",
            cost.iterations
        ));
    }
    let report = EffectReport { frame_reads: cost.reads, iterations: cost.iterations, tier: Tier::of(cost.reads) };
    Ok(Parsed { module, info, report })
}

/// Ce que coûte un bloc, par pixel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Cost {
    reads: u64,
    iterations: u64,
}

impl Cost {
    fn add(self, o: Cost) -> Cost {
        Cost { reads: self.reads.saturating_add(o.reads), iterations: self.iterations.saturating_add(o.iterations) }
    }
    fn max(self, o: Cost) -> Cost {
        Cost { reads: self.reads.max(o.reads), iterations: self.iterations.max(o.iterations) }
    }
    fn times(self, n: u64) -> Cost {
        Cost { reads: self.reads.saturating_mul(n), iterations: self.iterations.saturating_mul(n) }
    }
}

/// Parcourt l'IR de naga depuis `fx_fs` : les fonctions appelées sont dépliées, une branche
/// compte pour la plus chère, une boucle multiplie son corps par sa borne.
struct CostWalk<'a> {
    module: &'a naga::Module,
    src: &'a str,
    by_function: std::collections::HashMap<naga::Handle<naga::Function>, Cost>,
}

impl CostWalk<'_> {
    fn block(&mut self, func: &naga::Function, block: &naga::Block) -> Result<Cost, String> {
        let mut total = Cost::default();
        for (stmt, span) in block.span_iter() {
            let c = match stmt {
                naga::Statement::Emit(range) => {
                    let reads = range
                        .clone()
                        .filter(|&h| {
                            matches!(
                                func.expressions[h],
                                naga::Expression::ImageSample { .. } | naga::Expression::ImageLoad { .. }
                            )
                        })
                        .count() as u64;
                    Cost { reads, iterations: 0 }
                }
                naga::Statement::Block(b) => self.block(func, b)?,
                naga::Statement::If { accept, reject, .. } => {
                    self.block(func, accept)?.max(self.block(func, reject)?)
                }
                naga::Statement::Switch { cases, .. } => {
                    let mut worst = Cost::default();
                    for case in cases {
                        worst = worst.max(self.block(func, &case.body)?);
                    }
                    worst
                }
                naga::Statement::Loop { body, continuing, break_if } => {
                    let trips = self.trip_count(func, block, body, continuing, *break_if, *span)?;
                    let inner = self.block(func, body)?.add(self.block(func, continuing)?);
                    Cost { reads: inner.reads, iterations: inner.iterations.saturating_add(1) }.times(trips)
                }
                naga::Statement::Call { function, .. } => self.function(*function)?,
                _ => Cost::default(),
            };
            total = total.add(c);
        }
        Ok(total)
    }

    fn function(&mut self, h: naga::Handle<naga::Function>) -> Result<Cost, String> {
        if let Some(c) = self.by_function.get(&h) {
            return Ok(*c);
        }
        let module = self.module;
        let f = &module.functions[h];
        let c = self.block(f, &f.body)?;
        self.by_function.insert(h, c);
        Ok(c)
    }

    fn refuse(&self, span: naga::Span, why: &str) -> String {
        let line = span.location(self.src).line_number;
        format!(
            "the loop at line {line} {why}. Write it as `for (var i = 0; i < 16; i++)`: a counter \
             initialised with a literal, compared to a literal, stepped by a literal in the loop header"
        )
    }

    /// Le nombre de tours d'une boucle `for` à bornes littérales. Toute autre forme est refusée :
    /// une boucle qui ne finit pas bloque le GPU jusqu'au TDR.
    fn trip_count(
        &self,
        func: &naga::Function,
        parent: &naga::Block,
        body: &naga::Block,
        continuing: &naga::Block,
        break_if: Option<naga::Handle<naga::Expression>>,
        span: naga::Span,
    ) -> Result<u64, String> {
        use naga::{BinaryOperator as B, Expression as E, Statement as S};
        if break_if.is_some() {
            return Err(self.refuse(span, "ends on `break if`"));
        }
        // Le `for` de WGSL devient `loop { if (cond) {} else { break; } … continuing { pas } }`.
        let guard = body.iter().find(|s| !matches!(s, S::Emit(_)));
        let Some(S::If { condition, accept, reject }) = guard else {
            return Err(self.refuse(span, "has no bound in its header"));
        };
        if !accept.is_empty() || reject.len() != 1 || !matches!(reject.iter().next(), Some(S::Break)) {
            return Err(self.refuse(span, "has no bound in its header"));
        }
        let E::Binary { op, left, right } = func.expressions[*condition] else {
            return Err(self.refuse(span, "is not bounded by a comparison"));
        };
        let (var, bound, op) = match (self.counter(func, left), self.literal(func, right)) {
            (Some(v), Some(k)) => (v, k, op),
            _ => match (self.literal(func, left), self.counter(func, right)) {
                // `N > i` se lit `i < N`.
                (Some(k), Some(v)) => (v, k, match op {
                    B::Greater => B::Less,
                    B::GreaterEqual => B::LessEqual,
                    B::Less => B::Greater,
                    B::LessEqual => B::GreaterEqual,
                    other => other,
                }),
                _ => return Err(self.refuse(span, "is not bounded by a literal")),
            },
        };
        let init = func.local_variables[var]
            .init
            .and_then(|h| self.literal(func, h))
            .or_else(|| self.store_before(func, parent, var))
            .ok_or_else(|| self.refuse(span, "starts its counter from a value that is not a literal"))?;
        let step = self
            .step(func, continuing, var)
            .ok_or_else(|| self.refuse(span, "does not step its counter by a literal in its header"))?;
        if Self::writes(func, body, var) {
            return Err(self.refuse(span, "changes its counter inside its body"));
        }
        let trips = match op {
            B::Less if step > 0.0 => ((bound - init) / step).ceil(),
            B::LessEqual if step > 0.0 => ((bound - init) / step).floor() + 1.0,
            B::Greater if step < 0.0 => ((init - bound) / -step).ceil(),
            B::GreaterEqual if step < 0.0 => ((init - bound) / -step).floor() + 1.0,
            _ => return Err(self.refuse(span, "never reaches its bound")),
        };
        Ok(trips.max(0.0) as u64)
    }

    /// `Load(LocalVariable(v))` : le compteur d'une condition.
    fn counter(&self, func: &naga::Function, h: naga::Handle<naga::Expression>) -> Option<naga::Handle<naga::LocalVariable>> {
        match func.expressions[h] {
            naga::Expression::Load { pointer } => match func.expressions[pointer] {
                naga::Expression::LocalVariable(v) => Some(v),
                _ => None,
            },
            _ => None,
        }
    }

    /// Un littéral, directement ou par une constante de module.
    fn literal(&self, func: &naga::Function, h: naga::Handle<naga::Expression>) -> Option<f64> {
        let lit = |e: &naga::Expression| match *e {
            naga::Expression::Literal(l) => match l {
                naga::Literal::I32(v) => Some(v as f64),
                naga::Literal::U32(v) => Some(v as f64),
                naga::Literal::F32(v) => Some(v as f64),
                naga::Literal::I64(v) => Some(v as f64),
                naga::Literal::U64(v) => Some(v as f64),
                naga::Literal::F64(v) => Some(v),
                naga::Literal::AbstractInt(v) => Some(v as f64),
                naga::Literal::AbstractFloat(v) => Some(v),
                _ => None,
            },
            _ => None,
        };
        match &func.expressions[h] {
            naga::Expression::Constant(c) => lit(&self.module.global_expressions[self.module.constants[*c].init]),
            e => lit(e),
        }
    }

    /// Le dernier `Store` d'un littéral dans `var` qui précède une boucle dans son bloc.
    fn store_before(&self, func: &naga::Function, parent: &naga::Block, var: naga::Handle<naga::LocalVariable>) -> Option<f64> {
        let mut found = None;
        for s in parent.iter() {
            match s {
                naga::Statement::Store { pointer, value }
                    if matches!(func.expressions[*pointer], naga::Expression::LocalVariable(v) if v == var) =>
                {
                    found = self.literal(func, *value);
                }
                naga::Statement::Loop { .. } => {}
                _ => {}
            }
        }
        found
    }

    /// Le pas du compteur dans `continuing` : `i = i + k` ou `i = i - k`, `k` littéral.
    fn step(&self, func: &naga::Function, continuing: &naga::Block, var: naga::Handle<naga::LocalVariable>) -> Option<f64> {
        continuing.iter().find_map(|s| match s {
            naga::Statement::Store { pointer, value }
                if matches!(func.expressions[*pointer], naga::Expression::LocalVariable(v) if v == var) =>
            {
                match func.expressions[*value] {
                    naga::Expression::Binary { op, left, right } if self.counter(func, left) == Some(var) => {
                        let k = self.literal(func, right)?;
                        match op {
                            naga::BinaryOperator::Add => Some(k),
                            naga::BinaryOperator::Subtract => Some(-k),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        })
    }

    /// Le corps écrit-il le compteur (directement, ou en passant son pointeur à une fonction) ?
    fn writes(func: &naga::Function, block: &naga::Block, var: naga::Handle<naga::LocalVariable>) -> bool {
        let is_var = |h: naga::Handle<naga::Expression>| {
            matches!(func.expressions[h], naga::Expression::LocalVariable(v) if v == var)
        };
        block.iter().any(|s| match s {
            naga::Statement::Store { pointer, .. } => is_var(*pointer),
            naga::Statement::Call { arguments, .. } => arguments.iter().any(|&a| is_var(a)),
            naga::Statement::Block(b) => Self::writes(func, b, var),
            naga::Statement::If { accept, reject, .. } => Self::writes(func, accept, var) || Self::writes(func, reject, var),
            naga::Statement::Switch { cases, .. } => cases.iter().any(|c| Self::writes(func, &c.body, var)),
            naga::Statement::Loop { body, continuing, .. } => {
                Self::writes(func, body, var) || Self::writes(func, continuing, var)
            }
            _ => false,
        })
    }
}

/// HLSL SM 5.0 pour D3D11/FXC. naga 24 est la dernière version dont la sortie HLSL passe FXC :
/// à partir de la 25 elle déclare les samplers dans un tas D3D12 (`register(s0, space0)`), que
/// `ps_5_0` refuse (X3721). Ne pas monter naga sans faire passer ce module.
pub fn to_hlsl(parsed: &Parsed) -> Result<String, String> {
    let options = naga::back::hlsl::Options {
        shader_model: naga::back::hlsl::ShaderModel::V5_0,
        ..Default::default()
    };
    let mut out = String::new();
    naga::back::hlsl::Writer::new(&mut out, &options)
        .write(&parsed.module, &parsed.info, None)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

/// Clé du cache des shaders compilés : ce qui change le binaire, et rien d'autre.
pub fn source_hash(effect: &SceneShaderEffect) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    effect.contract.hash(&mut h);
    effect.code.hash(&mut h);
    for p in &effect.params {
        p.name.hash(&mut h);
        p.ty.hash(&mut h);
    }
    h.finish()
}

/// Les régions actives au temps source `t`, dans l'ordre où elles s'appliquent : par lane, puis
/// dans l'ordre de la scène.
pub fn active_effects(effects: &[SceneShaderEffect], t: f32) -> Vec<&SceneShaderEffect> {
    let t = t as f64;
    let mut active: Vec<_> = effects.iter().filter(|e| t >= e.start_sec && t < e.end_sec).collect();
    active.sort_by_key(|e| e.lane);
    active
}

/// La boîte englobante (x, y, w, h) d'un quad, en uv de sortie : `ctx.footage` sous un tilt 3D.
pub fn bounds(corners: &[[f32; 2]; 4]) -> [f32; 4] {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for [x, y] in corners {
        (x0, y0, x1, y1) = (x0.min(*x), y0.min(*y), x1.max(*x), y1.max(*y));
    }
    [x0, y0, x1 - x0, y1 - y0]
}

/// `ctx.region` : progression sur la pilule entière en temps programme quand l'app la donne, sur
/// le fragment en temps source sinon.
fn region(effect: &SceneShaderEffect, source_t: f32, programme_t: f32) -> [f32; 4] {
    let (start, end, now) = match (effect.programme_start_sec, effect.programme_end_sec) {
        (Some(s), Some(e)) => (s, e, programme_t as f64),
        _ => (effect.start_sec, effect.end_sec, source_t as f64),
    };
    let span = (end - start).max(1e-6);
    let since = (now - start).max(0.0);
    [(since / span).clamp(0.0, 1.0) as f32, span as f32, since as f32, 0.0]
}

pub fn context(
    effect: &SceneShaderEffect,
    source_t: f32,
    programme_t: f32,
    resolution: [f32; 2],
    inputs: &FxFrameInputs,
) -> FxContext {
    let mut clicks = [[0.0; 4]; MAX_CLICKS];
    for (slot, ([x, y], age)) in clicks.iter_mut().zip(&inputs.clicks) {
        *slot = [*x, *y, *age, 1.0];
    }
    FxContext {
        resolution,
        time: programme_t,
        aspect: resolution[0] / resolution[1].max(1.0),
        region: region(effect, source_t, programme_t),
        cursor: inputs.cursor.map_or([0.0; 4], |[x, y]| [x, y, 1.0, 0.0]),
        clicks,
        footage: inputs.footage,
        camera: inputs.camera.unwrap_or([0.0; 4]),
    }
}

/// Les paramètres de la région, rangés comme la struct `FxParams` générée (règles d'alignement de
/// WGSL), en mots de 4 octets. Une valeur absente vaut 0 : l'app envoie déjà les défauts.
pub fn pack_params(effect: &SceneShaderEffect) -> Vec<f32> {
    let mut words: Vec<f32> = Vec::new();
    let mut offset = 0usize;
    for p in &effect.params {
        let (size, align) = p.ty.size_align();
        offset = offset.div_ceil(align) * align;
        let end = offset + size;
        words.resize(end / 4, 0.0);
        let value: Vec<f32> = match effect.values.get(&p.name) {
            Some(ShaderParamValue::Scalar(v)) => vec![*v],
            Some(ShaderParamValue::Vector(v)) => v.clone(),
            None => Vec::new(),
        };
        for (i, v) in value.into_iter().take(size / 4).enumerate() {
            words[offset / 4 + i] = v;
        }
        offset = end;
    }
    words.resize(offset.div_ceil(16).max(1) * 4, 0.0);
    words
}

/// Le bytecode DXBC des deux étages, prêt pour `CreateVertexShader` / `CreatePixelShader`.
#[cfg(windows)]
pub struct CompiledFx {
    pub vs: Vec<u8>,
    pub ps: Vec<u8>,
}

#[cfg(windows)]
pub fn compile_dxbc(code: &str, params: &[ShaderParamDecl]) -> Result<CompiledFx, String> {
    let parsed = analyze(code, params)?;
    let hlsl = to_hlsl(&parsed)?;
    Ok(CompiledFx {
        vs: d3d_compile(&hlsl, "fx_vs", "vs_5_0")?,
        ps: d3d_compile(&hlsl, "fx_fs", "ps_5_0")?,
    })
}

#[cfg(windows)]
fn d3d_compile(hlsl: &str, entry: &str, profile: &str) -> Result<Vec<u8>, String> {
    use windows::core::PCSTR;
    use windows::Win32::Graphics::Direct3D::Fxc::{D3DCompile, D3DCOMPILE_OPTIMIZATION_LEVEL3};
    use windows::Win32::Graphics::Direct3D::ID3DBlob;

    fn bytes(blob: &ID3DBlob) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())
                .to_vec()
        }
    }
    let (entry_z, profile_z) = (format!("{entry}\0"), format!("{profile}\0"));
    let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    let compiled = unsafe {
        D3DCompile(
            hlsl.as_ptr().cast(),
            hlsl.len(),
            PCSTR(b"effect.hlsl\0".as_ptr()),
            None,
            None,
            PCSTR(entry_z.as_ptr()),
            PCSTR(profile_z.as_ptr()),
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    match (compiled, code) {
        (Ok(()), Some(code)) => Ok(bytes(&code)),
        (result, _) => {
            let log = errors.as_ref().map(bytes).unwrap_or_default();
            Err(format!("{entry}: {result:?}\n{}", String::from_utf8_lossy(&log)))
        }
    }
}

/// Compile sur un thread à part ; le résultat arrive par le canal. naga + FXC prennent ~10 ms
/// pour un effet de 60 lignes, soit une frame de preview : jamais sur le thread de rendu.
#[cfg(windows)]
pub fn spawn_compile(effect: &SceneShaderEffect) -> std::sync::mpsc::Receiver<Result<CompiledFx, String>> {
    let (tx, rx) = std::sync::mpsc::channel();
    let (code, params) = (effect.code.clone(), effect.params.clone());
    std::thread::Builder::new()
        .name("shader-fx".into())
        .spawn(move || {
            let _ = tx.send(compile_dxbc(&code, &params));
        })
        .expect("thread shader-fx");
    rx
}

/// Ce que l'outil de l'agent demande avant d'accepter un code : la validation, la compilation
/// réelle pour le backend de la machine quand elle existe, et le palier.
pub fn check_for_agent(code: &str, params: &[ShaderParamDecl]) -> Result<EffectReport, String> {
    let report = analyze(code, params)?.report;
    #[cfg(windows)]
    compile_dxbc(code, params)?;
    Ok(report)
}

/// `check_for_agent`, les paramètres reçus en JSON (`[{ "name", "type" }, …]`) depuis l'addon.
pub fn check_for_agent_json(code: &str, params_json: &str) -> Result<EffectReport, String> {
    let params: Vec<ShaderParamDecl> =
        serde_json::from_str(params_json).map_err(|e| format!("params: {e}"))?;
    check_for_agent(code, &params)
}

/// Un effet tel que les tests et les bancs le construisent.
#[cfg(test)]
pub(crate) fn test_effect(code: &str, params: Vec<ShaderParamDecl>, values: &[(&str, ShaderParamValue)]) -> SceneShaderEffect {
    SceneShaderEffect {
        clip_index: Some(0),
        start_sec: 1.0,
        end_sec: 3.0,
        programme_start_sec: None,
        programme_end_sec: None,
        target: ShaderEffectTarget::Image,
        lane: 0,
        contract: CONTRACT,
        code: code.into(),
        params,
        values: values.iter().map(|(k, v)| (k.to_string(), v.clone())).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decl(name: &str, ty: ShaderParamType) -> ShaderParamDecl {
        ShaderParamDecl { name: name.into(), ty }
    }

    const VIGNETTE: &str = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    let c = frame(uv);\n    let d = distance(uv, vec2<f32>(0.5));\n    return vec4<f32>(c.rgb * (1.0 - d * params.strength * ctx.region.x), c.a);\n}\n";

    #[test]
    fn a_valid_effect_becomes_hlsl_that_d3d11_accepts() {
        let parsed = analyze(VIGNETTE, &[decl("strength", ShaderParamType::Number)]).expect("analyze");
        let hlsl = to_hlsl(&parsed).expect("translate");
        assert!(hlsl.contains("fx_vs") && hlsl.contains("fx_fs"));
        // Les registres que les backends lient : t0, s1, b2 (ctx), b3 (params), sans `space`.
        for reg in ["register(t0)", "register(s1)", "register(b2)", "register(b3)"] {
            assert!(hlsl.contains(reg), "{reg} missing:\n{hlsl}");
        }
        assert!(!hlsl.contains("space"), "FXC ps_5_0 refuse les espaces de registres");
    }

    #[test]
    fn an_error_points_at_the_line_the_agent_wrote() {
        let broken = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    return frame(uv) * nope;\n}\n";
        let err = analyze(broken, &[]).err().expect("must fail");
        assert!(err.contains("nope") && err.contains(":2:"), "{err}");
    }

    #[test]
    fn an_effect_without_the_entry_function_is_refused() {
        assert!(analyze("fn other() {}", &[]).is_err());
    }

    #[test]
    fn parameter_names_are_checked() {
        for bad in ["Strength", "2x", "unused", "a-b"] {
            assert!(analyze(VIGNETTE, &[decl(bad, ShaderParamType::Number)]).is_err(), "{bad}");
        }
        let twice = [decl("a", ShaderParamType::Number), decl("a", ShaderParamType::Color)];
        assert!(analyze(VIGNETTE, &twice).is_err());
    }

    /// Le packer et la struct générée doivent placer chaque paramètre au même octet : un écart lirait
    /// une couleur là où l'app a écrit une force.
    #[test]
    fn the_packer_matches_the_offsets_naga_gives_the_generated_struct() {
        let params = vec![
            decl("a", ShaderParamType::Number),
            decl("tint", ShaderParamType::Color),
            decl("on", ShaderParamType::Toggle),
            decl("at", ShaderParamType::Point),
            decl("b", ShaderParamType::Number),
        ];
        let code = "fn effect(uv: vec2<f32>) -> vec4<f32> { return frame(uv) * params.a + params.tint * params.on + vec4<f32>(params.at, params.b, 0.0); }";
        let parsed = analyze(code, &params).expect("analyze");
        let ty = parsed
            .module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("FxParams"))
            .expect("FxParams")
            .1;
        let naga::TypeInner::Struct { members, span } = &ty.inner else { panic!("struct") };
        let values = [
            ("a", ShaderParamValue::Scalar(1.0)),
            ("tint", ShaderParamValue::Vector(vec![2.0, 3.0, 4.0, 5.0])),
            ("on", ShaderParamValue::Scalar(6.0)),
            ("at", ShaderParamValue::Vector(vec![7.0, 8.0])),
            ("b", ShaderParamValue::Scalar(9.0)),
        ];
        let words = pack_params(&test_effect(code, params, &values));
        assert_eq!(words.len() * 4, (*span as usize).div_ceil(16) * 16);
        let first = |name: &str| members.iter().find(|m| m.name.as_deref() == Some(name)).unwrap().offset as usize / 4;
        assert_eq!(words[first("a")], 1.0);
        assert_eq!(&words[first("tint")..first("tint") + 4], &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(words[first("on")], 6.0);
        assert_eq!(&words[first("at")..first("at") + 2], &[7.0, 8.0]);
        assert_eq!(words[first("b")], 9.0);
    }

    #[test]
    fn the_context_matches_its_wgsl_struct() {
        let parsed = analyze(VIGNETTE, &[decl("strength", ShaderParamType::Number)]).unwrap();
        let ty = parsed.module.types.iter().find(|(_, t)| t.name.as_deref() == Some("FxContext")).unwrap().1;
        let naga::TypeInner::Struct { members, span } = &ty.inner else { panic!("struct") };
        assert_eq!(std::mem::size_of::<FxContext>(), *span as usize);
        let offset = |name: &str| members.iter().find(|m| m.name.as_deref() == Some(name)).unwrap().offset as usize;
        let c = FxContext::default();
        let base = &c as *const FxContext as usize;
        assert_eq!(offset("region"), &c.region as *const _ as usize - base);
        assert_eq!(offset("clicks"), &c.clicks as *const _ as usize - base);
        assert_eq!(offset("camera"), &c.camera as *const _ as usize - base);
    }

    #[test]
    fn a_for_loop_with_literal_bounds_is_counted() {
        let code = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    var acc = vec4<f32>(0.0);\n    for (var i = 0; i < 16; i++) {\n        acc += frame(uv + vec2<f32>(f32(i) * 0.001, 0.0));\n    }\n    return acc / 16.0;\n}\n";
        let r = analyze(code, &[]).expect("analyze").report;
        assert_eq!((r.frame_reads, r.iterations, r.tier), (16, 16, Tier::Medium));
    }

    #[test]
    fn nested_loops_multiply_and_called_functions_are_unfolded() {
        let code = "fn tap(uv: vec2<f32>) -> vec4<f32> { return frame(uv) + frame(uv * 0.5); }\nfn effect(uv: vec2<f32>) -> vec4<f32> {\n    var acc = vec4<f32>(0.0);\n    for (var y = 0u; y <= 7u; y += 1u) {\n        for (var x = 8; x > 0; x--) {\n            acc += tap(uv);\n        }\n    }\n    return acc;\n}\n";
        let r = analyze(code, &[]).expect("analyze").report;
        assert_eq!(r.frame_reads, 8 * 8 * 2);
        assert_eq!(r.iterations, 8 * (8 + 1));
        assert_eq!(r.tier, Tier::Heavy);
    }

    #[test]
    fn a_branch_counts_for_the_dearer_side() {
        let code = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    if (uv.x > 0.5) { return frame(uv) + frame(uv) + frame(uv); }\n    return frame(uv);\n}\n";
        assert_eq!(analyze(code, &[]).unwrap().report.frame_reads, 4, "branche chère + suite");
    }

    #[test]
    fn loops_without_a_literal_bound_are_refused_with_their_line() {
        let refused = [
            ("fn effect(uv: vec2<f32>) -> vec4<f32> {\n    var i = 0;\n    loop {\n        i++;\n        if (i > 4) { break; }\n    }\n    return frame(uv);\n}\n", 3),
            ("fn effect(uv: vec2<f32>) -> vec4<f32> {\n    var i = 0;\n    while (i < 4) {\n        i++;\n    }\n    return frame(uv);\n}\n", 3),
            ("fn effect(uv: vec2<f32>) -> vec4<f32> {\n    let n = i32(uv.x * 100.0);\n    for (var i = 0; i < n; i++) {}\n    return frame(uv);\n}\n", 3),
            ("fn effect(uv: vec2<f32>) -> vec4<f32> {\n    for (var i = 0; i < 4; i++) { i = 0; }\n    return frame(uv);\n}\n", 2),
            ("fn effect(uv: vec2<f32>) -> vec4<f32> {\n    for (var i = 0; i < 4; i--) {}\n    return frame(uv);\n}\n", 2),
        ];
        for (code, line) in refused {
            let err = analyze(code, &[]).err().unwrap_or_else(|| panic!("accepted: {code}"));
            assert!(err.contains(&format!("line {line}")) && err.contains("for (var i = 0"), "{err}");
        }
    }

    #[test]
    fn a_module_constant_is_a_literal_bound() {
        let code = "const TAPS = 12;\nfn effect(uv: vec2<f32>) -> vec4<f32> {\n    var acc = vec4<f32>(0.0);\n    for (var i = 0; i < TAPS; i += 3) { acc += frame(uv); }\n    return acc;\n}\n";
        assert_eq!(analyze(code, &[]).expect("analyze").report.iterations, 4);
    }

    #[test]
    fn the_safety_ceiling_refuses_runaway_work() {
        let code = "fn effect(uv: vec2<f32>) -> vec4<f32> {\n    var a = 0.0;\n    for (var i = 0; i < 100; i++) { for (var j = 0; j < 100; j++) { a += 1.0; } }\n    return frame(uv) * a;\n}\n";
        let err = analyze(code, &[]).err().expect("must fail");
        assert!(err.contains("10100") && err.contains("4096"), "{err}");
    }

    #[test]
    fn the_ripple_of_the_spec_is_light() {
        let code = include_str!("../../../docs/shader-effects.md")
            .split("```wgsl\n")
            .nth(2)
            .and_then(|s| s.split("```").next())
            .expect("l'exemple C.6");
        let params = vec![
            decl("strength", ShaderParamType::Number),
            decl("radius", ShaderParamType::Number),
            decl("duration", ShaderParamType::Number),
        ];
        let r = analyze(code, &params).expect("l'exemple de la spec compile").report;
        assert_eq!((r.frame_reads, r.tier), (1, Tier::Light));
    }

    #[test]
    fn the_active_regions_and_their_progress() {
        let mut a = test_effect(VIGNETTE, vec![], &[]);
        a.lane = 1;
        let b = test_effect(VIGNETTE, vec![], &[]);
        let effects = [a, b];
        assert!(active_effects(&effects, 0.5).is_empty());
        let active = active_effects(&effects, 2.0);
        assert_eq!(active.iter().map(|e| e.lane).collect::<Vec<_>>(), [0, 1], "par lane");
        assert!(active_effects(&effects, 3.0).is_empty(), "fin exclusive");

        let ctx = context(&effects[1], 2.0, 10.0, [1920.0, 1080.0], &FxFrameInputs::default());
        assert_eq!(ctx.region, [0.5, 2.0, 1.0, 0.0], "fragment, temps source");
        let mut whole = effects[1].clone();
        (whole.programme_start_sec, whole.programme_end_sec) = (Some(8.0), Some(12.0));
        let ctx = context(&whole, 2.0, 10.0, [1920.0, 1080.0], &FxFrameInputs::default());
        assert_eq!(ctx.region, [0.5, 4.0, 2.0, 0.0], "pilule entière, temps programme");
        assert_eq!((ctx.cursor[2], ctx.clicks[0][3], ctx.camera), (0.0, 0.0, [0.0; 4]));

        let inputs = FxFrameInputs {
            cursor: Some([0.25, 0.5]),
            clicks: vec![([0.3, 0.6], 0.2), ([0.1, 0.1], 1.5)],
            footage: [0.1, 0.1, 0.8, 0.8],
            camera: Some([0.7, 0.7, 0.2, 0.2]),
        };
        let ctx = context(&whole, 2.0, 10.0, [1920.0, 1080.0], &inputs);
        assert_eq!(ctx.cursor, [0.25, 0.5, 1.0, 0.0]);
        assert_eq!(ctx.clicks[..3], [[0.3, 0.6, 0.2, 1.0], [0.1, 0.1, 1.5, 1.0], [0.0; 4]]);
        assert_eq!((ctx.footage, ctx.camera), ([0.1, 0.1, 0.8, 0.8], [0.7, 0.7, 0.2, 0.2]));
    }

    #[test]
    fn the_cache_key_follows_code_and_layout_not_values() {
        let p = vec![decl("strength", ShaderParamType::Number)];
        let a = test_effect(VIGNETTE, p.clone(), &[("strength", ShaderParamValue::Scalar(1.0))]);
        let mut b = test_effect(VIGNETTE, p, &[("strength", ShaderParamValue::Scalar(2.0))]);
        assert_eq!(source_hash(&a), source_hash(&b), "une valeur ne recompile rien");
        b.params[0].ty = ShaderParamType::Color;
        assert_ne!(source_hash(&a), source_hash(&b), "un layout différent est un autre binaire");
    }

    #[cfg(windows)]
    #[test]
    fn fxc_compiles_both_stages() {
        let c = compile_dxbc(VIGNETTE, &[decl("strength", ShaderParamType::Number)]).expect("compile");
        assert!(!c.vs.is_empty() && !c.ps.is_empty());
    }
}
