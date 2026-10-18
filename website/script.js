/* ==========================================================================
   INSIDE DIABETES — Complete Cinematic Medical Documentary & AI Engine
   Complete Human Body Anatomy + Realistic Internal Organs + WebGL Raycaster
   ========================================================================== */

'use strict';

const PREFERS_REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Embedded authentic model data trained on Pima Indians Diabetes Dataset
let MODEL_DATA = {"model_name": "Explainable Feedforward ANN (MLP)", "architecture": "8 Inputs -> 16 Hidden (ReLU) -> 8 Hidden (ReLU) -> 1 Output (Sigmoid)", "dataset": "Pima Indians Diabetes Database (NIDDK)", "total_samples": 768, "metrics": {"accuracy": 0.6948051948051948, "precision": 0.5660377358490566, "recall": 0.5555555555555556, "f1_score": 0.5607476635514018, "roc_auc": 0.8087037037037037, "confusion_matrix": [[77, 23], [24, 30]]}, "feature_names": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"], "feature_specs": {"Pregnancies": {"unit": "count", "min": 0, "max": 20, "default": 1, "step": 1}, "Glucose": {"unit": "mg/dL", "min": 40, "max": 300, "default": 100, "step": 1}, "BloodPressure": {"unit": "mm Hg", "min": 30, "max": 180, "default": 72, "step": 1}, "SkinThickness": {"unit": "mm", "min": 5, "max": 99, "default": 23, "step": 1}, "Insulin": {"unit": "\u00b5U/mL", "min": 10, "max": 850, "default": 85, "step": 1}, "BMI": {"unit": "kg/m\u00b2", "min": 12.0, "max": 67.0, "default": 26.5, "step": 0.1}, "DiabetesPedigreeFunction": {"unit": "score", "min": 0.05, "max": 2.5, "default": 0.38, "step": 0.01}, "Age": {"unit": "years", "min": 18, "max": 100, "default": 33, "step": 1}}, "medians": {"Pregnancies": 3.0, "Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "Insulin": 125.0, "BMI": 32.4, "DiabetesPedigreeFunction": 0.3825, "Age": 29.0}, "scaler_mean": [3.8192182410423454, 121.67100977198697, 72.1400651465798, 29.042345276872965, 137.70521172638436, 32.44820846905537, 0.477428338762215, 33.36644951140065], "scaler_std": [3.311448216586372, 29.979350764655706, 12.2651192609104, 8.884611281688818, 78.70059996835236, 6.81856216148434, 0.33003119077476745, 11.82379790254742], "weights": [[[-0.09573028269522642, 0.4927257103292036, 0.07646700776660192, 0.10661745591000087, -0.9452043551531554, -0.22264725370750035, -0.5489492901246851, 0.6305598643260814, -0.10372588241051589, -0.23775473688535279, -0.23062913387465594, 0.6011600454646845, 0.2593713533051942, -0.4996659897452234, -0.42794143973229126, -0.04921481941651534], [-0.15923748664247162, -0.24960287699119454, -0.26961041587136053, -0.15704941475523154, 0.3592160605366379, -0.6464758723193211, 0.0723766922902154, 0.16035278851105844, -0.063910276744989, 0.10640245728182886, -0.5534162096827394, 0.32990123144957045, -0.20825979855224025, -0.28139366136535676, 0.47649250474647137, -0.1895366264138792], [-0.2947900425085086, 0.21947816503610884, 0.27249409908423694, 0.04947349444811089, -0.19195373428237064, 0.16930034969961458, -0.062169112083572994, -0.0688114341519922, -0.43780243816359893, -0.07085558055702172, -0.6653583787517279, 0.4989145114307045, -0.7029515825962962, 0.3053133212552336, -0.410988898413168, 0.15185462251597515], [0.03675217472325295, -0.19031694666494126, 0.5438980697994544, 0.23429570260540983, 0.0094985558467588, 0.3016995413026437, 0.2414598079340453, 0.3581573973597034, -0.22074725779544943, -0.5241887123951112, -0.5983657315562688, -0.3146622439302157, -0.36624107201463846, -0.16725543663257494, 0.37953060581092984, -0.3829427793548017], [-0.6442714286873722, -0.012709523602832212, -1.03331610082391, 0.3145386355433315, -0.23365834593217863, 0.6684593421792497, 0.23532098951226874, -0.25890232937861596, -0.4388582563550268, 0.27340720582165495, 0.14378313481312804, 0.5403243703660698, -0.19019181908544694, -0.16243517812562455, -0.13434510600716457, -0.300242222272244], [0.3169226000620149, 0.27153365160420223, -0.2358528813586526, -0.6386486853955309, 0.004314728198522511, -0.2915105694243393, 0.4447543705863714, 0.24982830152933483, 0.2615813686224154, -0.236127815034459, -0.701555514051535, 0.36488647584715905, 0.062425854482133045, 0.00063144697830162, 0.3135373565685123, 0.00011006340957537652], [-0.14719147700771662, -0.19465230578588086, -0.563169178540695, -0.19038051653372037, -0.19487008713155618, -0.010500097816150057, -0.08312587095790493, -0.13501399850850335, 0.020610638049153637, -0.30010270252648524, -0.41494542967413045, 0.37380410253495394, 0.08647160705825935, -0.34193058090890316, -0.0873847847153121, -0.1361879518949183], [0.7657768025914983, 0.28580510406419346, -0.1563675251053182, 0.3804012360834045, 0.4301062795231917, -0.6057879003940386, 0.7504790731141627, -0.06749624134039903, 0.3951880182206557, 0.34254027550885496, -0.4271713987707891, -0.14460784265830087, -0.43131013834381315, 0.29315305126296426, 0.3860771354186898, 0.7043608530914905]], [[-0.5148566073171076, -0.032890149575846184, 0.4830029010650214, 0.2873399107980687, -0.6289820446660285, 0.2528148885463922, 0.8114272856198654, -0.5803872717583786], [-0.12992829693202074, -0.02624624352438069, 0.5596638234533873, -0.3763120322867602, 0.30699157562119084, 0.3472419876934477, -0.135456621252442, 0.387698144408789], [-0.31498453440103963, 0.3788480838746646, 0.4599411004216003, -0.15239913443527917, -0.8348925605919882, 0.6665955457168045, -0.056222679549284775, -0.3045191905166451], [-0.3935779269246108, 0.14074916927631823, 0.3302454399114516, -0.4167377267155049, 0.05650350687961153, -0.16740122686205106, 0.3791759156641812, -0.09323705231791052], [0.42571422962583466, -0.3321310249798425, 0.3000190919044779, -0.016076333925249514, 0.13521022925026432, -0.5468608841786952, 0.2895889958627449, 0.5900708188012842], [-0.4986533563657734, 0.5477593620696303, 0.46179723477554707, 0.45352406664354256, -0.24279539027980263, -0.038487882018565504, -0.07288260885826521, 0.12888750392683543], [0.5826932718401235, 0.02319617567719332, -0.21478862160247425, -0.4403694151586796, 0.5268881519993998, 0.23603645991715577, 0.4821743641498148, 0.44501604405529094], [0.4946591518258714, -0.44357691892685375, -0.34318548331911, 0.28951032588567555, 0.6578244959940731, -0.270465752546574, -1.0071080989191417, 0.5030331060234212], [-0.3048450051722195, -0.2978928703968873, 0.18057421550081945, 0.11240841362887158, 0.2969237793018567, -0.015229233377617285, 0.23398433198923305, -0.28733515253943137], [0.1831911599712039, 0.0057311425645537595, -0.07811690705720649, 0.2758266053574201, 0.42934194820409227, -0.17403876913890318, -0.6712371332411831, 0.08441483330626022], [-0.2709721667546705, -0.04921687788963452, 0.7018632346981072, 0.4050198677151439, 0.17807282303386798, 0.374195138225979, 0.5475028337931831, -0.13505373367070622], [0.1816558246300077, -0.05684556475267971, -0.48674390427739345, 0.12257087334615718, -0.1341617905978958, -0.8265233333713562, 0.3043285743992602, -0.21067832264060185], [0.7558366815712569, 0.34371599308753026, 0.23700737155188326, -0.27081054777815117, -0.6031505577875979, 0.3622377777508223, -0.2687342026525299, 0.6263541439581409], [0.45522876000951995, 0.4306028767259923, -0.15280912225535573, 0.27911519850229977, 0.2949109253686679, -0.12313545056938882, -0.4361739455295071, 0.09497013668770642], [0.7190871740822601, 0.026065068095311558, 0.1622058003309486, -1.019505672230159, 0.4366070373114558, 0.5075857004799502, -0.6438656331622777, 0.21271150925650092], [0.1781242944122034, 0.49699189510549907, 0.5310002029203352, 0.7571059236137517, -0.30155178434618723, 0.08287341470304357, 0.6695336634871606, 0.11144559789165367]], [[0.840850635532329], [-0.26646118471564234], [-0.4907480582193615], [-0.9509247644345521], [0.3040322537665253], [-1.0208068690669685], [-0.5124205300293488], [0.14708107305943427]]], "biases": [[-0.3846406203047214, -0.3883159982553722, -0.22929161523943317, -0.340393510383411, -0.2778827897067043, 0.183961697187783, 0.548403314608678, -0.37975761991985363, -0.2727124846231563, 0.06468177141668276, -0.12005591102072903, 0.2989467221054629, 0.6539865263995097, -0.44397870390565003, 0.4995056374878751, -0.4026896266692574], [0.4891056294649858, 0.554457851735247, 0.2453932379473335, -0.07923953562039307, 0.123516236916706, 0.32195017364120676, 0.26723463628118305, 0.2266982694941854], [-0.34580217006817415]], "expected_value": 0.35465005937623123, "global_importance_pct": {"Pregnancies": 11.4, "Glucose": 20.3, "BloodPressure": 6.8, "SkinThickness": 6.9, "Insulin": 15.1, "BMI": 15.1, "DiabetesPedigreeFunction": 8.3, "Age": 16.0}};

/* ==========================================================================
   1. GLOBAL THREE.JS SCENE SETUP
   ========================================================================== */
const canvas = document.getElementById('webgl-canvas');
const renderer = new THREE.WebGLRenderer({
  canvas: canvas,
  antialias: true,
  alpha: true,
  powerPreference: 'high-performance'
});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // Cap at 1.5 for performance
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
// Soft shadows (kept but with reduced resolution)
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x020408, 0.016);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 22);
const cameraTarget = new THREE.Vector3(1.8, 0, 0);

// Cinematic Multi-Point Biological Lighting with Soft Realistic Shadows
scene.add(new THREE.AmbientLight(0x1a2638, 1.3));

// Key Directional Light for Anatomical Models with Soft Shadows
const keyLight = new THREE.DirectionalLight(0xfff5ea, 2.3);
keyLight.position.set(12, 18, 16);
keyLight.castShadow = true;
keyLight.shadow.mapSize.width = 1024;   // Reduced from 2048 for performance
keyLight.shadow.mapSize.height = 1024;
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 70;
keyLight.shadow.camera.left = -16;
keyLight.shadow.camera.right = 16;
keyLight.shadow.camera.top = 16;
keyLight.shadow.camera.bottom = -16;
keyLight.shadow.bias = -0.0004;
keyLight.shadow.radius = 1.8;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x38bdf8, 1.1);
fillLight.position.set(-16, -8, -6);
scene.add(fillLight);

const biologicalRimLight = new THREE.PointLight(0xdc2626, 3.8, 60);
biologicalRimLight.position.set(4, 0, 8);
scene.add(biologicalRimLight);

// Dedicated Bloodstream Lumen Projector (Produces striking soft shadows inside vessel)
const bloodstreamSpot = new THREE.SpotLight(0xffecd2, 4.2, 60, Math.PI / 3.0, 0.45, 1.2);
bloodstreamSpot.position.set(0, -22, -22);
bloodstreamSpot.target.position.set(0, -22, 12);
// No shadow on bloodstream spot — too expensive with many transparent cells
scene.add(bloodstreamSpot);
scene.add(bloodstreamSpot.target);

// Subtle biological rim backlight for cellular translucency
const bloodstreamBacklight = new THREE.DirectionalLight(0x38bdf8, 0.85);
bloodstreamBacklight.position.set(-8, -20, 15);
scene.add(bloodstreamBacklight);

const cellProbeLight = new THREE.PointLight(0x06b6d4, 2.5, 45);
cellProbeLight.position.set(0, -75, 6);
// No shadow — point light shadow is expensive and not visible at this depth
scene.add(cellProbeLight);

/* ==========================================================================
   2. PROCEDURAL TEXTURES, SHADERS & CELLULAR BIOMECHANICS
   ========================================================================== */
// 1. Endothelial Diffuse Texture (Vascular Cobblestone)
function createEndothelialTexture() {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');

  ctx.fillStyle = '#220306';
  ctx.fillRect(0, 0, size, size);

  const numCells = 80;
  for (let i = 0; i < numCells; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const radius = 18 + Math.random() * 26;
    const grad = ctx.createRadialGradient(x, y, 2, x, y, radius);
    grad.addColorStop(0, 'rgba(175, 28, 48, 0.7)');
    grad.addColorStop(0.7, 'rgba(110, 15, 28, 0.5)');
    grad.addColorStop(1, 'rgba(30, 4, 8, 0.95)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // Endothelial nucleus
    ctx.fillStyle = 'rgba(240, 90, 110, 0.55)';
    ctx.beginPath();
    ctx.ellipse(x + (Math.random() - 0.5) * 6, y + (Math.random() - 0.5) * 6, 4, 2.5, Math.random() * Math.PI, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(c);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 12);
  return texture;
}

// 2. Endothelial Height / Bump Map for Realistic Vascular Wall Depth
function createEndothelialBumpTexture() {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < 90; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 16 + Math.random() * 24;
    const grad = ctx.createRadialGradient(x, y, 1, x, y, r);
    grad.addColorStop(0, '#ffffff');
    grad.addColorStop(0.65, '#999999');
    grad.addColorStop(1, '#505050');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(c);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 12);
  return texture;
}

// 3. Erythrocyte Membrane Micro-wrinkle Bump Map
function createErythrocyteBumpTexture() {
  const size = 256;
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < 700; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 1.2 + Math.random() * 2.5;
    ctx.fillStyle = Math.random() > 0.5 ? '#9e9e9e' : '#686868';
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(c);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(3, 3);
  return texture;
}

// 4. White Blood Cell (Leukocyte) Microvilli & Ruffled Surface Bump Map
function createLeukocyteBumpTexture() {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < 1800; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 2 + Math.random() * 3.8;
    const grad = ctx.createRadialGradient(x, y, 0.5, x, y, r);
    grad.addColorStop(0, '#ffffff');
    grad.addColorStop(0.5, '#aaaaaa');
    grad.addColorStop(1, '#666666');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(c);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 4);
  return texture;
}

const endothelialTexture     = createEndothelialTexture();
const endothelialBumpTexture = createEndothelialBumpTexture();
const erythrocyteBumpTexture = createErythrocyteBumpTexture();
const leukocyteBumpTexture   = createLeukocyteBumpTexture();

// Evans-Fung biomechanical erythrocyte lathe geometry
function createErythrocyteGeometry(radius = 1.15, radialSegments = 32, pointsCount = 28) {
  const points = [];
  for (let i = 0; i <= pointsCount; i++) {
    const u = -Math.PI / 2 + (Math.PI * i / pointsCount);
    const r = Math.cos(u) * radius;
    const y = Math.sin(u) * (0.16 + 0.38 * (Math.cos(u) * Math.cos(u))) * radius;
    points.push(new THREE.Vector2(Math.max(0, r), y));
  }
  return new THREE.LatheGeometry(points, radialSegments);
}

const erythrocyteGeom = createErythrocyteGeometry(1.15, 24, 20); // Fewer segments for performance

// Erythrocyte Material (MeshStandard for performance — still beautiful)
const erythrocyteMaterial = new THREE.MeshStandardMaterial({
  color: 0xba181b,
  emissive: 0x3d0205,
  roughness: 0.28,
  metalness: 0.12,
  bumpMap: erythrocyteBumpTexture,
  bumpScale: 0.035
});

// Leukocyte Outer Material (opaque ivory — no transmission = much faster)
const leukocyteOuterMaterial = new THREE.MeshStandardMaterial({
  color: 0xeef2f6,
  emissive: 0x1a2233,
  roughness: 0.38,
  metalness: 0.05,
  bumpMap: leukocyteBumpTexture,
  bumpScale: 0.07
});

// Neutrophil Multilobed Chromatin Nucleus
const leukocyteNucleusMaterial = new THREE.MeshStandardMaterial({
  color: 0x4c1d95,
  emissive: 0x2e1065,
  roughness: 0.36,
  metalness: 0.12
});

/* ==========================================================================
   3. 3D BIOLOGICAL SCENE GROUPS & RAYCASTER REGISTRATION
   ========================================================================== */
const interactiveObjects = [];

function registerBioObject(mesh, name, desc) {
  mesh.userData.bioInfo = { name, desc };
  interactiveObjects.push(mesh);
}

// ── GROUP 1: COMPLETE HUMAN BODY ANATOMY & ORGANS (HERO) ────────────────
const heroGroup = new THREE.Group();
heroGroup.position.set(3.8, 0, 0); // Framed cleanly on right side
scene.add(heroGroup);

// Fresnel Holographic Body Shell Shader
const bodyShellMaterial = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  uniforms: {
    uColor: { value: new THREE.Color(0x38bdf8) },
    uTime:  { value: 0.0 }
  },
  vertexShader: `
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    void main() {
      vNormal = normalize(normalMatrix * normal);
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      vViewPosition = -mvPosition.xyz;
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    uniform vec3 uColor;
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    void main() {
      vec3 normal = normalize(vNormal);
      vec3 viewDir = normalize(vViewPosition);
      float fresnel = pow(1.0 - abs(dot(normal, viewDir)), 2.8);
      float alpha = clamp(0.12 + fresnel * 0.75, 0.0, 0.85);
      vec3 col = mix(uColor * 0.35, vec3(1.0), fresnel * 0.4);
      gl_FragColor = vec4(col, alpha);
    }
  `
});

// Build Complete Anatomical Human Body (Head, Torso, Arms, Legs, Organs)
let heartMeshRef = null;
let leftLungRef = null;
let rightLungRef = null;

function createCompleteHumanBody() {
  const body = new THREE.Group();

  // ── A. ANATOMICAL SILHOUETTE SHELL ─────────────────────────────
  // 1. Head / Cranium
  const headGeom = new THREE.SphereGeometry(0.85, 24, 20);
  headGeom.scale(1.0, 1.25, 1.1);
  const headMesh = new THREE.Mesh(headGeom, bodyShellMaterial);
  headMesh.position.set(0, 6.2, 0);
  registerBioObject(headMesh, 'Human Cranium & Cerebrum', 'Houses the central nervous system and hypothalamic glucose-sensing nuclei that modulate autonomic tone.');
  body.add(headMesh);

  // 2. Neck
  const neckMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.48, 0.85, 16), bodyShellMaterial);
  neckMesh.position.set(0, 5.0, 0);
  body.add(neckMesh);

  // 3. Thorax & Ribcage Shell
  const thoraxGeom = new THREE.CylinderGeometry(1.45, 1.15, 2.6, 20);
  thoraxGeom.scale(1.35, 1.0, 0.85);
  const thoraxMesh = new THREE.Mesh(thoraxGeom, bodyShellMaterial);
  thoraxMesh.position.set(0, 3.4, 0);
  body.add(thoraxMesh);

  // 4. Abdomen & Pelvis Shell
  const abdomenGeom = new THREE.CylinderGeometry(1.15, 1.35, 2.4, 20);
  abdomenGeom.scale(1.3, 1.0, 0.88);
  const abdomenMesh = new THREE.Mesh(abdomenGeom, bodyShellMaterial);
  abdomenMesh.position.set(0, 1.0, 0);
  body.add(abdomenMesh);

  // 5. Shoulders & Arms
  const shoulderMat = bodyShellMaterial;
  const upperArmGeom = new THREE.CylinderGeometry(0.32, 0.28, 2.0, 14);
  const forearmGeom  = new THREE.CylinderGeometry(0.26, 0.22, 1.9, 14);
  const handGeom     = new THREE.SphereGeometry(0.24, 12, 12);
  handGeom.scale(0.8, 1.4, 0.5);

  // Left Arm
  const leftUpperArm = new THREE.Mesh(upperArmGeom, shoulderMat);
  leftUpperArm.position.set(-2.0, 3.2, 0);
  leftUpperArm.rotation.z = -0.15;
  body.add(leftUpperArm);

  const leftForearm = new THREE.Mesh(forearmGeom, shoulderMat);
  leftForearm.position.set(-2.3, 1.4, 0);
  leftForearm.rotation.z = -0.1;
  body.add(leftForearm);

  const leftHand = new THREE.Mesh(handGeom, shoulderMat);
  leftHand.position.set(-2.45, 0.2, 0);
  body.add(leftHand);

  // Right Arm
  const rightUpperArm = new THREE.Mesh(upperArmGeom, shoulderMat);
  rightUpperArm.position.set(2.0, 3.2, 0);
  rightUpperArm.rotation.z = 0.15;
  body.add(rightUpperArm);

  const rightForearm = new THREE.Mesh(forearmGeom, shoulderMat);
  rightForearm.position.set(2.3, 1.4, 0);
  rightForearm.rotation.z = 0.1;
  body.add(rightForearm);

  const rightHand = new THREE.Mesh(handGeom, shoulderMat);
  rightHand.position.set(2.45, 0.2, 0);
  body.add(rightHand);

  // 6. Pelvis & Lower Limbs
  const thighGeom = new THREE.CylinderGeometry(0.55, 0.42, 3.2, 16);
  const calfGeom  = new THREE.CylinderGeometry(0.40, 0.30, 3.1, 16);
  const footGeom  = new THREE.BoxGeometry(0.5, 0.35, 1.1);

  // Left Leg
  const leftThigh = new THREE.Mesh(thighGeom, shoulderMat);
  leftThigh.position.set(-0.85, -1.8, 0);
  body.add(leftThigh);

  const leftCalf = new THREE.Mesh(calfGeom, shoulderMat);
  leftCalf.position.set(-0.85, -4.8, 0);
  body.add(leftCalf);

  const leftFoot = new THREE.Mesh(footGeom, shoulderMat);
  leftFoot.position.set(-0.85, -6.5, 0.3);
  body.add(leftFoot);

  // Right Leg
  const rightThigh = new THREE.Mesh(thighGeom, shoulderMat);
  rightThigh.position.set(0.85, -1.8, 0);
  body.add(rightThigh);

  const rightCalf = new THREE.Mesh(calfGeom, shoulderMat);
  rightCalf.position.set(0.85, -4.8, 0);
  body.add(rightCalf);

  const rightFoot = new THREE.Mesh(footGeom, shoulderMat);
  rightFoot.position.set(0.85, -6.5, 0.3);
  body.add(rightFoot);

  // ── B. INTERNAL ORGANS ─────────────────────────────────────────

  // 1. BEATING HEART (Thoracic Cavity, left of midline)
  const heartGroup = new THREE.Group();
  heartGroup.position.set(-0.35, 3.3, 0.45);

  const heartMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    emissive: 0x991b1b,
    roughness: 0.35
  });

  // Anatomical Heart Shape: Cone for ventricles + spheres for atria
  const ventricleMesh = new THREE.Mesh(new THREE.ConeGeometry(0.45, 0.9, 16), heartMat);
  ventricleMesh.rotation.z = Math.PI - 0.25;
  ventricleMesh.rotation.y = 0.2;
  heartGroup.add(ventricleMesh);

  const atriumL = new THREE.Mesh(new THREE.SphereGeometry(0.28, 12, 12), heartMat);
  atriumL.position.set(-0.18, 0.35, 0);
  heartGroup.add(atriumL);

  const atriumR = new THREE.Mesh(new THREE.SphereGeometry(0.26, 12, 12), heartMat);
  atriumR.position.set(0.18, 0.32, 0);
  heartGroup.add(atriumR);

  registerBioObject(heartGroup, 'Human Heart (Cardiac Muscle)', 'Four-chambered muscular pump beating ~72 times/minute, propelling pressurized blood into the aorta and systemic vascular circuit.');
  body.add(heartGroup);
  heartMeshRef = heartGroup;

  // 2. LUNGS (Bilateral Pulmonary Lobes)
  const lungMat = new THREE.MeshStandardMaterial({
    color: 0x64748b,
    emissive: 0x1e293b,
    roughness: 0.5,
    transparent: true,
    opacity: 0.72
  });

  const rightLungGeom = new THREE.ConeGeometry(0.65, 1.8, 14);
  rightLungGeom.scale(1.1, 1.0, 0.8);
  const rightLung = new THREE.Mesh(rightLungGeom, lungMat);
  rightLung.position.set(1.0, 3.4, 0.2);
  rightLung.rotation.z = -0.08;
  registerBioObject(rightLung, 'Right Lung (3 Pulmonary Lobes)', 'Facilitates gas exchange across alveolar-capillary membranes, oxygenating venous blood returned from systemic tissues.');
  body.add(rightLung);
  rightLungRef = rightLung;

  const leftLungGeom = new THREE.ConeGeometry(0.6, 1.7, 14);
  leftLungGeom.scale(1.0, 1.0, 0.75);
  const leftLung = new THREE.Mesh(leftLungGeom, lungMat);
  leftLung.position.set(-1.0, 3.4, 0.2);
  leftLung.rotation.z = 0.08;
  registerBioObject(leftLung, 'Left Lung (2 Lobes & Cardiac Notch)', 'Provides oxygenation with an anatomical medial cardiac notch accommodating the ventricular apex.');
  body.add(leftLung);
  leftLungRef = leftLung;

  // 3. LIVER (Right Upper Quadrant)
  const liverGeom = new THREE.ConeGeometry(0.9, 1.4, 16);
  liverGeom.scale(1.6, 0.75, 0.9);
  const liverMesh = new THREE.Mesh(liverGeom, new THREE.MeshStandardMaterial({
    color: 0x854d0e,
    emissive: 0x451a03,
    roughness: 0.45
  }));
  liverMesh.position.set(0.85, 1.9, 0.35);
  liverMesh.rotation.z = -0.35;
  liverMesh.rotation.y = 0.4;
  registerBioObject(liverMesh, 'Liver (Hepatic Metabolic Center)', 'Major regulator of carbohydrate metabolism: synthesizes glycogen, performs gluconeogenesis during fasting, and clears insulin from portal blood.');
  body.add(liverMesh);

  // 4. STOMACH (Left Epigastrium)
  const stomachCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.3, 2.3, 0.3),
    new THREE.Vector3(-0.9, 2.0, 0.4),
    new THREE.Vector3(-0.8, 1.3, 0.4),
    new THREE.Vector3(-0.2, 1.2, 0.35)
  ]);
  const stomachMesh = new THREE.Mesh(
    new THREE.TubeGeometry(stomachCurve, 20, 0.35, 12, false),
    new THREE.MeshStandardMaterial({ color: 0xa16207, roughness: 0.5 })
  );
  registerBioObject(stomachMesh, 'Stomach (Gastric Reservoir)', 'Digests ingested macronutrients and releases glucose precursors into the duodenum, stimulating incretin hormone secretion.');
  body.add(stomachMesh);

  // 5. PANCREAS (Retroperitoneal behind stomach)
  const pancreasSpine = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.6, 1.0, 0.2),   // Head in duodenal loop
    new THREE.Vector3(0.0, 1.15, 0.15), // Body
    new THREE.Vector3(-0.9, 1.35, 0.05) // Tail
  ]);
  const bodyPancreas = new THREE.Mesh(
    new THREE.TubeGeometry(pancreasSpine, 24, 0.22, 10, false),
    new THREE.MeshStandardMaterial({ color: 0x10b981, emissive: 0x065f46, roughness: 0.35 })
  );
  registerBioObject(bodyPancreas, 'Pancreas (Endocrine Islets)', 'Houses the Islets of Langerhans. Beta cells directly detect blood sugar and secrete insulin to drive glucose clearance into muscle and fat.');
  body.add(bodyPancreas);

  // 6. KIDNEYS (Bilateral Retroperitoneal)
  const kidneyGeom = new THREE.SphereGeometry(0.3, 12, 12);
  kidneyGeom.scale(0.8, 1.3, 0.6);
  const kidneyMat = new THREE.MeshStandardMaterial({ color: 0x991b1b, emissive: 0x450a0a, roughness: 0.4 });

  const rightKidney = new THREE.Mesh(kidneyGeom, kidneyMat);
  rightKidney.position.set(0.9, 0.9, -0.2);
  registerBioObject(rightKidney, 'Right Kidney (Renal Filtration)', 'Filters plasma metabolites. Chronic hyperglycemia damages renal glomeruli, leading to diabetic nephropathy.');
  body.add(rightKidney);

  const leftKidney = new THREE.Mesh(kidneyGeom, kidneyMat);
  leftKidney.position.set(-0.9, 1.0, -0.2);
  registerBioObject(leftKidney, 'Left Kidney (Renal Glomeruli)', 'Maintains fluid and electrolyte homeostasis; high blood glucose over years leads to progressive glomerular hyperfiltration injury.');
  body.add(leftKidney);

  // ── C. EXTENSIVE VASCULAR NETWORK (HEAD TO FEET) ────────────────

  const vasRedMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, emissive: 0x991b1b, roughness: 0.3 });
  const venBlueMat= new THREE.MeshStandardMaterial({ color: 0x0284c7, emissive: 0x0369a1, roughness: 0.35 });

  // Aorta & Major Arterial Highway
  const aortaCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.25, 3.6, 0.35), // Aortic arch
    new THREE.Vector3(0.0, 3.8, 0.25),
    new THREE.Vector3(0.15, 2.8, 0.0),   // Thoracic aorta
    new THREE.Vector3(0.1, 1.2, 0.0),    // Abdominal aorta
    new THREE.Vector3(0.0, 0.1, 0.05),   // Bifurcation
    new THREE.Vector3(0.45, -1.0, 0.1),  // Right iliac
    new THREE.Vector3(0.85, -2.5, 0.1),  // Right femoral
    new THREE.Vector3(0.85, -4.5, 0.05), // Right tibial
    new THREE.Vector3(0.85, -6.2, 0.1)
  ]);
  const aortaMesh = new THREE.Mesh(new THREE.TubeGeometry(aortaCurve, 45, 0.12, 8, false), vasRedMat);
  registerBioObject(aortaMesh, 'Aorta & Systemic Arterial Highway', 'The primary high-pressure conduit transporting oxygenated blood and circulating glucose from the left ventricle into systemic organs.');
  body.add(aortaMesh);

  // Left Iliac & Femoral Artery
  const leftLegArtery = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.0, 0.1, 0.05),
    new THREE.Vector3(-0.45, -1.0, 0.1),
    new THREE.Vector3(-0.85, -2.5, 0.1),
    new THREE.Vector3(-0.85, -4.5, 0.05),
    new THREE.Vector3(-0.85, -6.2, 0.1)
  ]);
  body.add(new THREE.Mesh(new THREE.TubeGeometry(leftLegArtery, 35, 0.10, 8, false), vasRedMat));

  // Carotid Arteries (Up to Head)
  const leftCarotid = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.15, 3.7, 0.2),
    new THREE.Vector3(-0.35, 4.8, 0.1),
    new THREE.Vector3(-0.4, 6.0, 0.1)
  ]);
  const rightCarotid = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.1, 3.7, 0.2),
    new THREE.Vector3(0.35, 4.8, 0.1),
    new THREE.Vector3(0.4, 6.0, 0.1)
  ]);
  body.add(new THREE.Mesh(new THREE.TubeGeometry(leftCarotid, 20, 0.07, 8, false), vasRedMat));
  body.add(new THREE.Mesh(new THREE.TubeGeometry(rightCarotid, 20, 0.07, 8, false), vasRedMat));

  // Arm Arteries (Subclavian to Brachial)
  const leftArmArtery = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.25, 3.7, 0.2),
    new THREE.Vector3(-1.3, 4.0, 0.1),
    new THREE.Vector3(-2.0, 3.2, 0.0),
    new THREE.Vector3(-2.3, 1.4, 0.0),
    new THREE.Vector3(-2.45, 0.2, 0.0)
  ]);
  const rightArmArtery = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.15, 3.7, 0.2),
    new THREE.Vector3(1.3, 4.0, 0.1),
    new THREE.Vector3(2.0, 3.2, 0.0),
    new THREE.Vector3(2.3, 1.4, 0.0),
    new THREE.Vector3(2.45, 0.2, 0.0)
  ]);
  body.add(new THREE.Mesh(new THREE.TubeGeometry(leftArmArtery, 28, 0.07, 8, false), vasRedMat));
  body.add(new THREE.Mesh(new THREE.TubeGeometry(rightArmArtery, 28, 0.07, 8, false), vasRedMat));

  // Inferior & Superior Vena Cava
  const venaCavaCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.05, 5.0, 0.1),
    new THREE.Vector3(-0.05, 3.5, 0.2),
    new THREE.Vector3(-0.08, 1.8, 0.05),
    new THREE.Vector3(-0.08, 0.2, 0.05)
  ]);
  const venaMesh = new THREE.Mesh(new THREE.TubeGeometry(venaCavaCurve, 25, 0.13, 8, false), venBlueMat);
  registerBioObject(venaMesh, 'Vena Cava (Venous Return Conduit)', 'Transports deoxygenated blood from abdominal organs, brain, and lower extremities back to the right atrium.');
  body.add(venaMesh);

  // Micro-arterial branching network (over 70 biological curves)
  const branchMat = new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending });
  for (let i = 0; i < 75; i++) {
    const yVal = 5.5 - Math.random() * 11.5;
    const start = new THREE.Vector3((Math.random() - 0.5) * 0.6, yVal, (Math.random() - 0.5) * 0.3);
    const end = new THREE.Vector3(
      start.x + (Math.random() - 0.5) * 3.8,
      start.y + (Math.random() - 0.5) * 2.2,
      start.z + (Math.random() - 0.5) * 1.5
    );
    const mid = new THREE.Vector3((start.x + end.x) * 0.5, (start.y + end.y) * 0.5, (start.z + end.z) * 0.5);
    const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
    body.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(10)), branchMat));
  }

  return body;
}

const completeHumanBody = createCompleteHumanBody();
heroGroup.add(completeHumanBody);

// Enable shadow casting and receiving across all anatomical structures
function applyGlobalShadowProperties(group) {
  if (!group) return;
  group.traverse(child => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });
}

applyGlobalShadowProperties(completeHumanBody);

// Floating interstitial particles
const particleCount = 180;
const particleGeom = new THREE.BufferGeometry();
const particlePos = new Float32Array(particleCount * 3);
for (let i = 0; i < particleCount * 3; i += 3) {
  particlePos[i] = (Math.random() - 0.5) * 32;
  particlePos[i + 1] = (Math.random() - 0.5) * 32;
  particlePos[i + 2] = (Math.random() - 0.5) * 20;
}
particleGeom.setAttribute('position', new THREE.BufferAttribute(particlePos, 3));
const floatingParticles = new THREE.Points(particleGeom, new THREE.PointsMaterial({
  color: 0xfca5a5, size: 0.14, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending
}));
scene.add(floatingParticles);

// ── GROUP 2: BLOODSTREAM (ERYTHROCYTES, LEUKOCYTES & D-GLUCOSE) ────────
const bloodstreamGroup = new THREE.Group();
bloodstreamGroup.position.set(0, -22, 0);
scene.add(bloodstreamGroup);

const vesselRadius = 5.2;
const vesselLength = 48;
const vesselGeom = new THREE.CylinderGeometry(vesselRadius, vesselRadius, vesselLength, 48, 1, true);
const vesselMat = new THREE.MeshStandardMaterial({
  map: endothelialTexture,
  bumpMap: endothelialBumpTexture,
  bumpScale: 0.16,
  side: THREE.BackSide,
  roughness: 0.42,
  metalness: 0.08,
  color: 0x991b1b
});
const vesselMesh = new THREE.Mesh(vesselGeom, vesselMat);
vesselMesh.rotation.x = Math.PI / 2;
vesselMesh.receiveShadow = true; // Inner wall receives dramatic soft shadows from cells
// Do not register vesselMesh in interactiveObjects so empty plasma does not block cell popups
bloodstreamGroup.add(vesselMesh);

// 1. Erythrocytes (Red Blood Cells) with Shadows & Depth
const rbcList = [];
const numRBCs = 26;  // Reduced for performance
for (let i = 0; i < numRBCs; i++) {
  const mesh = new THREE.Mesh(erythrocyteGeom, erythrocyteMaterial);
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  const angle = Math.random() * Math.PI * 2;
  const dist = Math.random() * (vesselRadius - 1.4);
  mesh.position.set(Math.cos(angle) * dist, Math.sin(angle) * dist, (Math.random() - 0.5) * vesselLength * 0.95);
  mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
  mesh.scale.setScalar(0.72 + Math.random() * 0.32);

  mesh.userData = {
    vz: 0.06 + Math.random() * 0.07,
    rotX: (Math.random() - 0.5) * 0.03,
    rotY: (Math.random() - 0.5) * 0.03,
    rotZ: (Math.random() - 0.5) * 0.03
  };
  registerBioObject(mesh, 'Red Blood Cell (Erythrocyte)', 'Biconcave disc (~7.8 µm) containing hemoglobin. Its high flexibility enables transit through micro-capillaries.');
  bloodstreamGroup.add(mesh);
  rbcList.push(mesh);
}

// 2. White Blood Cells (Leukocytes / Neutrophils) with Multilobed Nuclei
// Perfectly scaled, slower rolling, with 100% responsive cursor hover popup
function createWhiteBloodCell() {
  const wbcGroup = new THREE.Group();

  const bioInfo = {
    name: 'White Blood Cell (Leukocyte / Neutrophil)',
    desc: 'Spherical immune sentinel (~12–15 µm) exhibiting endothelial rolling. Chronic hyperglycemia activates leukocytes, prompting vascular adhesion and inflammatory endothelial damage.'
  };

  // A. Outer perturbed ruffled spherical membrane (compact, authentic biological scale)
  const baseRadius = 0.95;
  const wbcGeom = new THREE.SphereGeometry(baseRadius, 28, 24);
  const pos = wbcGeom.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const px = pos.getX(i);
    const py = pos.getY(i);
    const pz = pos.getZ(i);
    const displacement = 1.0 + 0.045 * Math.sin(px * 5.0 + py * 3.5) + 0.028 * Math.cos(pz * 5.5);
    pos.setXYZ(i, px * displacement, py * displacement, pz * displacement);
  }
  wbcGeom.computeVertexNormals();
  wbcGeom.computeBoundingSphere();

  const outerMesh = new THREE.Mesh(wbcGeom, leukocyteOuterMaterial);
  outerMesh.castShadow = true;
  outerMesh.receiveShadow = true;
  outerMesh.userData.bioInfo = bioInfo;
  wbcGroup.add(outerMesh);

  // B. Generous invisible hit-sphere for effortless and reliable cursor raycasting
  const hitGeom = new THREE.SphereGeometry(baseRadius * 1.25, 12, 12);
  const hitMat = new THREE.MeshBasicMaterial({ visible: false });
  const hitMesh = new THREE.Mesh(hitGeom, hitMat);
  hitMesh.userData.bioInfo = bioInfo;
  wbcGroup.add(hitMesh);

  // C. Characteristic Multilobed Neutrophil Chromatin Nucleus (3-4 interconnected lobes)
  const nucleusGroup = new THREE.Group();
  const lobeGeom = new THREE.SphereGeometry(0.24, 14, 14);
  lobeGeom.scale(1.1, 0.85, 0.9);

  const lobePositions = [
    new THREE.Vector3(-0.24, 0.15, 0.08),
    new THREE.Vector3(0.22, 0.20, -0.08),
    new THREE.Vector3(0.15, -0.23, 0.10),
    new THREE.Vector3(-0.19, -0.16, -0.10)
  ];

  lobePositions.forEach(p => {
    const lobe = new THREE.Mesh(lobeGeom, leukocyteNucleusMaterial);
    lobe.position.copy(p);
    lobe.castShadow = true;
    lobe.receiveShadow = true;
    lobe.userData.bioInfo = bioInfo;
    nucleusGroup.add(lobe);
  });

  // Nuclear chromatin connecting bridges
  const bridgeMat = new THREE.MeshStandardMaterial({ color: 0x3b0764, roughness: 0.4 });
  for (let i = 0; i < lobePositions.length - 1; i++) {
    const p1 = lobePositions[i];
    const p2 = lobePositions[i + 1];
    const dist = p1.distanceTo(p2);
    const bridgeGeom = new THREE.CylinderGeometry(0.065, 0.065, dist, 8);
    const bridge = new THREE.Mesh(bridgeGeom, bridgeMat);
    bridge.position.copy(p1).add(p2).multiplyScalar(0.5);
    bridge.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), p2.clone().sub(p1).normalize());
    bridge.castShadow = true;
    bridge.userData.bioInfo = bioInfo;
    nucleusGroup.add(bridge);
  }

  wbcGroup.add(nucleusGroup);

  // Register all interactive targets
  wbcGroup.userData.bioInfo = bioInfo;
  registerBioObject(outerMesh, bioInfo.name, bioInfo.desc);
  registerBioObject(hitMesh, bioInfo.name, bioInfo.desc);
  registerBioObject(wbcGroup, bioInfo.name, bioInfo.desc);

  return wbcGroup;
}

const wbcList = [];
const numWBCs = 8;   // Reduced for performance
for (let i = 0; i < numWBCs; i++) {
  const wbc = createWhiteBloodCell();
  const angle = Math.random() * Math.PI * 2;
  // Natural margination: WBCs roll along vessel wall periphery
  const wallDist = (vesselRadius - 1.1) - Math.random() * 0.5;
  wbc.position.set(Math.cos(angle) * wallDist, Math.sin(angle) * wallDist, (Math.random() - 0.5) * vesselLength * 0.95);
  // Significantly smaller scale — WBC ~1.6x RBC diameter (authentic)
  wbc.scale.setScalar(0.42 + Math.random() * 0.12);
  // IMPORTANT: Use Object.assign to PRESERVE bioInfo while adding animation data
  Object.assign(wbc.userData, {
    vz: 0.012 + Math.random() * 0.010, // Slow rolling speed (3x slower than RBCs)
    rollX: (Math.random() - 0.5) * 0.006,
    rollY: 0.005 + Math.random() * 0.006,
    phase: Math.random() * Math.PI * 2
  });
  bloodstreamGroup.add(wbc);
  wbcList.push(wbc);
}

// 3. Upgraded 3D D-Glucose Monosaccharide Molecules (Chair Pyranose Conformation)
function createGlucoseMolecule() {
  const molGroup = new THREE.Group();

  // Amber crystalline materials (MeshStandard for performance)
  const carbonMat = new THREE.MeshStandardMaterial({
    color: 0xf59e0b,
    roughness: 0.22,
    metalness: 0.18,
    emissive: 0xb45309,
    emissiveIntensity: 0.32
  });
  const oxygenMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    roughness: 0.24,
    metalness: 0.14,
    emissive: 0x9b1c1c,
    emissiveIntensity: 0.35
  });
  const hydroxylMat = new THREE.MeshStandardMaterial({
    color: 0xfde047,
    roughness: 0.25,
    metalness: 0.1,
    emissive: 0x92400e,
    emissiveIntensity: 0.22
  });
  const bondMat = new THREE.MeshStandardMaterial({
    color: 0xe2e8f0,
    roughness: 0.28,
    metalness: 0.35
  });

  const atomGeom = new THREE.SphereGeometry(0.12, 8, 8);
  const oGeom    = new THREE.SphereGeometry(0.14, 8, 8);
  const hGeom    = new THREE.SphereGeometry(0.08, 6, 6);
  const bondGeom = new THREE.CylinderGeometry(0.028, 0.028, 0.45, 8);

  // Stable 4C1 chair pyranose ring nodes (C1-C5 and ring oxygen O5)
  const ringNodes = [
    { x: 0.38,  y: 0.22,  z: 0.14,  type: 'C' }, // C1
    { x: 0.42,  y: -0.22, z: -0.12, type: 'C' }, // C2
    { x: -0.02, y: -0.38, z: 0.16,  type: 'C' }, // C3
    { x: -0.42, y: -0.20, z: -0.14, type: 'C' }, // C4
    { x: -0.38, y: 0.24,  z: 0.12,  type: 'C' }, // C5
    { x: 0.02,  y: 0.38,  z: -0.15, type: 'O' }  // O5 (Ring Oxygen)
  ];

  ringNodes.forEach((node, idx) => {
    const mesh = new THREE.Mesh(node.type === 'C' ? atomGeom : oGeom, node.type === 'C' ? carbonMat : oxygenMat);
    mesh.position.set(node.x, node.y, node.z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    molGroup.add(mesh);

    // Ring covalent bond
    const next = ringNodes[(idx + 1) % ringNodes.length];
    const bond = new THREE.Mesh(bondGeom, bondMat);
    const p1 = new THREE.Vector3(node.x, node.y, node.z);
    const p2 = new THREE.Vector3(next.x, next.y, next.z);
    bond.position.copy(p1).add(p2).multiplyScalar(0.5);
    bond.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), p2.clone().sub(p1).normalize());
    bond.scale.y = p1.distanceTo(p2) / 0.45;
    bond.castShadow = true;
    bond.receiveShadow = true;
    molGroup.add(bond);

    // Hydroxyl functional groups (-OH)
    if (node.type === 'C') {
      const ohDir = new THREE.Vector3(node.x, node.y, node.z).normalize().multiplyScalar(0.32);
      const ohMesh = new THREE.Mesh(hGeom, hydroxylMat);
      ohMesh.position.set(node.x + ohDir.x, node.y + ohDir.y, node.z + ohDir.z);
      ohMesh.castShadow = true;
      molGroup.add(ohMesh);

      const ohBond = new THREE.Mesh(bondGeom, bondMat);
      const op1 = new THREE.Vector3(node.x, node.y, node.z);
      const op2 = ohMesh.position;
      ohBond.position.copy(op1).add(op2).multiplyScalar(0.5);
      ohBond.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), op2.clone().sub(op1).normalize());
      ohBond.scale.y = op1.distanceTo(op2) / 0.45;
      ohBond.castShadow = true;
      molGroup.add(ohBond);
    }
  });

  // C6 Hydroxymethyl group (-CH2OH branching from C5)
  const c5 = ringNodes[4];
  const c6Pos = new THREE.Vector3(c5.x - 0.28, c5.y + 0.32, c5.z + 0.1);
  const c6Mesh = new THREE.Mesh(atomGeom, carbonMat);
  c6Mesh.position.copy(c6Pos);
  c6Mesh.castShadow = true;
  molGroup.add(c6Mesh);

  const c6Bond = new THREE.Mesh(bondGeom, bondMat);
  const cp1 = new THREE.Vector3(c5.x, c5.y, c5.z);
  c6Bond.position.copy(cp1).add(c6Pos).multiplyScalar(0.5);
  c6Bond.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), c6Pos.clone().sub(cp1).normalize());
  c6Bond.scale.y = cp1.distanceTo(c6Pos) / 0.45;
  c6Bond.castShadow = true;
  molGroup.add(c6Bond);

  const glucoseBioName = 'D-Glucose Molecule (C₆H₁₂O₆)';
  const glucoseBioDesc = 'Hexose monosaccharide in stable chair pyranose conformation. Persistent excess blood glucose non-enzymatically glycates hemoglobin (HbA1c) and degrades the microvascular endothelium.';

  // Invisible hit sphere for reliable raycasting on the tiny molecule
  const glucoseHitGeom = new THREE.SphereGeometry(0.55, 8, 8);
  const glucoseHitMat  = new THREE.MeshBasicMaterial({ visible: false });
  const glucoseHitMesh = new THREE.Mesh(glucoseHitGeom, glucoseHitMat);
  glucoseHitMesh.userData.bioInfo = { name: glucoseBioName, desc: glucoseBioDesc };
  molGroup.add(glucoseHitMesh);

  // Attach bioInfo to group AND register hit sphere
  molGroup.userData.bioInfo = { name: glucoseBioName, desc: glucoseBioDesc };
  registerBioObject(glucoseHitMesh, glucoseBioName, glucoseBioDesc);
  registerBioObject(molGroup,      glucoseBioName, glucoseBioDesc);
  return molGroup;
}

const glucoseMolecules = [];
const numGlucose = 30; // Reduced for performance
for (let i = 0; i < numGlucose; i++) {
  const g = createGlucoseMolecule();
  const angle = Math.random() * Math.PI * 2;
  const dist = Math.random() * (vesselRadius - 1.0);
  g.position.set(Math.cos(angle) * dist, Math.sin(angle) * dist, (Math.random() - 0.5) * vesselLength * 0.95);
  g.scale.setScalar(0.72 + Math.random() * 0.32);
  g.userData = {
    vz: 0.08 + Math.random() * 0.09,
    rotX: (Math.random() - 0.5) * 0.05,
    rotY: (Math.random() - 0.5) * 0.05
  };
  bloodstreamGroup.add(g);
  glucoseMolecules.push(g);
}

// ── GROUP 3: PANCREAS & ISLETS ──────────────────────────────────────────
const pancreasGroup = new THREE.Group();
pancreasGroup.position.set(0, -48, 0);
scene.add(pancreasGroup);

function createPancreasOrgan() {
  const organ = new THREE.Group();
  const pancreasMat = new THREE.MeshStandardMaterial({ color: 0xcaa07a, roughness: 0.55, metalness: 0.08 });

  const spinePoints = [
    new THREE.Vector3(3.2, -1.2, 0),
    new THREE.Vector3(3.0, 0.4, 0.2),
    new THREE.Vector3(1.8, 0.8, 0.4),
    new THREE.Vector3(0.0, 0.7, 0.2),
    new THREE.Vector3(-1.8, 0.9, -0.1),
    new THREE.Vector3(-3.4, 1.4, -0.4)
  ];
  const pancreasBodyMesh = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(spinePoints), 36, 1.1, 16, false), pancreasMat);
  registerBioObject(pancreasBodyMesh, 'Pancreas (Anatomical Organ)', 'Retroperitoneal gland with dual function: producing digestive enzymes and secreting blood sugar regulating hormones into the portal vein.');
  organ.add(pancreasBodyMesh);

  // Duodenal Loop
  const duodenalMesh = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
      new THREE.Vector3(2.4, 2.0, 0.1), new THREE.Vector3(4.2, 1.5, 0.2), new THREE.Vector3(4.6, -0.4, 0.1), new THREE.Vector3(4.0, -2.2, 0), new THREE.Vector3(2.0, -2.5, -0.2)
    ]), 28, 0.85, 14, false),
    new THREE.MeshStandardMaterial({ color: 0x9f5a43, roughness: 0.6 })
  );
  registerBioObject(duodenalMesh, 'Duodenum (C-Loop)', 'First section of the small intestine cradling the pancreatic head and receiving bile and pancreatic digestive secretions.');
  organ.add(duodenalMesh);

  // Pancreatic Duct
  const ductMesh = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
      new THREE.Vector3(3.1, -0.3, 0.2), new THREE.Vector3(1.8, 0.75, 0.45), new THREE.Vector3(0.0, 0.7, 0.25), new THREE.Vector3(-1.8, 0.9, -0.05), new THREE.Vector3(-3.2, 1.35, -0.35)
    ]), 30, 0.09, 8, false),
    new THREE.MeshStandardMaterial({ color: 0xbae6fd, emissive: 0x0284c7, roughness: 0.3 })
  );
  registerBioObject(ductMesh, 'Main Pancreatic Duct (Duct of Wirsung)', 'Central drainage conduit channeling bicarbonate and digestive enzymes toward the hepatopancreatic ampulla.');
  organ.add(ductMesh);

  // Splenic Artery
  const splenicPoints = [];
  for (let i = 0; i <= 24; i++) {
    const t = i / 24;
    splenicPoints.push(new THREE.Vector3(2.2 - t * 5.6, 1.3 + Math.sin(t * 12) * 0.25, 0.3 + Math.cos(t * 8) * 0.2));
  }
  const splenicMesh = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(splenicPoints), 32, 0.12, 8, false),
    new THREE.MeshStandardMaterial({ color: 0xdc2626, emissive: 0x991b1b, roughness: 0.3 })
  );
  registerBioObject(splenicMesh, 'Splenic Artery', 'Tortuous arterial branch off the celiac trunk that supplies the pancreatic body and tail before reaching the spleen.');
  organ.add(splenicMesh);

  return organ;
}

const pancreasOrgan = createPancreasOrgan();
pancreasGroup.add(pancreasOrgan);

// 3D Islet of Langerhans
function createIsletCluster() {
  const islet = new THREE.Group();
  islet.position.set(-0.8, 0.7, 2.5);

  const betaMat = new THREE.MeshStandardMaterial({ color: 0x10b981, emissive: 0x065f46, roughness: 0.35 });
  const alphaMat = new THREE.MeshStandardMaterial({ color: 0xf97316, emissive: 0x9a3412, roughness: 0.4 });
  const deltaMat = new THREE.MeshStandardMaterial({ color: 0xa855f7, emissive: 0x6b21a8, roughness: 0.4 });
  const cellGeom = new THREE.SphereGeometry(0.22, 12, 12);

  for (let i = 0; i < 50; i++) {
    const phi = Math.acos(-1 + (2 * i) / 50);
    const theta = Math.sqrt(50 * Math.PI) * phi;
    const r = 1.1 + (Math.random() - 0.5) * 0.3;

    const randType = Math.random();
    let mat = betaMat;
    let name = 'Pancreatic Beta (β) Cell';
    let desc = 'Produces and secretes insulin in response to rising blood glucose. In Type 1 diabetes, beta cells are targeted by autoimmune destruction.';
    if (randType > 0.85) {
      mat = deltaMat;
      name = 'Pancreatic Delta (δ) Cell';
      desc = 'Secretes somatostatin, an inhibitory peptide that exerts local paracrine restraint on both insulin and glucagon release.';
    } else if (randType > 0.70) {
      mat = alphaMat;
      name = 'Pancreatic Alpha (α) Cell';
      desc = 'Secretes glucagon during fasting to trigger hepatic glucose production, preventing hypoglycemia.';
    }

    const cellMesh = new THREE.Mesh(cellGeom, mat);
    cellMesh.position.set(r * Math.cos(theta) * Math.sin(phi), r * Math.sin(theta) * Math.sin(phi), r * Math.cos(phi));
    cellMesh.scale.setScalar(0.8 + Math.random() * 0.4);
    registerBioObject(cellMesh, name, desc);
    islet.add(cellMesh);
  }

  // Capillary rings
  const capMat = new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.7 });
  for (let c = 0; c < 5; c++) {
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(new THREE.EllipseCurve(0, 0, 1.35, 1.35, 0, 2 * Math.PI).getPoints(24)), capMat);
    line.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
    registerBioObject(line, 'Islet Endocrine Capillary', 'Fenestrated microvascular vessel that enables secreted insulin and glucagon to rapidly enter systemic circulation.');
    islet.add(line);
  }

  // Exocytosis granules
  const granuleCount = 30;
  const granuleGeom = new THREE.BufferGeometry();
  const granulePos = new Float32Array(granuleCount * 3);
  for (let g = 0; g < granuleCount * 3; g += 3) {
    granulePos[g] = (Math.random() - 0.5) * 2.2;
    granulePos[g + 1] = (Math.random() - 0.5) * 2.2;
    granulePos[g + 2] = (Math.random() - 0.5) * 2.2;
  }
  granuleGeom.setAttribute('position', new THREE.BufferAttribute(granulePos, 3));
  const granuleParticles = new THREE.Points(granuleGeom, new THREE.PointsMaterial({
    color: 0x38bdf8, size: 0.08, transparent: true, opacity: 0.85
  }));
  registerBioObject(granuleParticles, 'Insulin Secretory Granules', 'Dense-core micro-vesicles storing crystallized zinc-insulin hexamers awaiting calcium-triggered exocytosis.');
  islet.add(granuleParticles);

  islet.userData = { granules: granuleParticles };
  return islet;
}

const isletCluster = createIsletCluster();
pancreasGroup.add(isletCluster);
applyGlobalShadowProperties(pancreasGroup);

// ── GROUP 4: CELL MEMBRANE, RECEPTOR & GLUT4 ────────────────────────────
const cellMembraneGroup = new THREE.Group();
cellMembraneGroup.position.set(0, -75, 0);
scene.add(cellMembraneGroup);

function createLipidBilayer() {
  const membrane = new THREE.Group();
  const headGeom = new THREE.SphereGeometry(0.18, 10, 10);
  const headMatOuter = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.3, emissive: 0x0369a1 });
  const headMatInner = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.4 });

  const rows = 24;
  const cols = 24;
  const spacing = 0.44;
  const headsList = [];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = (c - cols / 2) * spacing;
      const z = (r - rows / 2) * spacing;

      const outerHead = new THREE.Mesh(headGeom, headMatOuter);
      outerHead.position.set(x, 0.45, z);
      membrane.add(outerHead);
      headsList.push(outerHead);

      const innerHead = new THREE.Mesh(headGeom, headMatInner);
      innerHead.position.set(x, -0.45, z);
      membrane.add(innerHead);
      headsList.push(innerHead);
    }
  }

  registerBioObject(membrane, 'Phospholipid Bilayer Membrane', 'The hydrophobic lipid barrier of target muscle/fat cells, preventing passive diffusion of hydrophilic polar glucose molecules.');
  membrane.userData = { heads: headsList };
  return membrane;
}

const lipidBilayer = createLipidBilayer();
cellMembraneGroup.add(lipidBilayer);

function createInsulinReceptor() {
  const ir = new THREE.Group();
  ir.position.set(-1.8, 0, 0);

  const alphaMat = new THREE.MeshStandardMaterial({ color: 0x06b6d4, roughness: 0.35, emissive: 0x0891b2 });
  const betaMat  = new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.4, emissive: 0x1d4ed8 });

  const alphaLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.14, 1.6, 12), alphaMat);
  alphaLeft.position.set(-0.35, 1.2, 0);
  alphaLeft.rotation.z = 0.3;
  ir.add(alphaLeft);

  const alphaRight = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.14, 1.6, 12), alphaMat);
  alphaRight.position.set(0.35, 1.2, 0);
  alphaRight.rotation.z = -0.3;
  ir.add(alphaRight);

  const betaLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.16, 2.0, 10), betaMat);
  betaLeft.position.set(-0.25, -0.6, 0);
  ir.add(betaLeft);

  const betaRight = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.16, 2.0, 10), betaMat);
  betaRight.position.set(0.25, -0.6, 0);
  ir.add(betaRight);

  const dockedInsulin = createGlucoseMolecule();
  dockedInsulin.position.set(0, 1.8, 0);
  dockedInsulin.scale.setScalar(0.7);
  ir.add(dockedInsulin);

  registerBioObject(ir, 'Tyrosine Kinase Insulin Receptor', 'Transmembrane heterotetramer (2α, 2β). Insulin binding induces autophosphorylation of β-subunits, launching the intracellular Akt cascade.');
  return ir;
}

const insulinReceptor = createInsulinReceptor();
cellMembraneGroup.add(insulinReceptor);

function createGlut4Channel() {
  const glut = new THREE.Group();
  glut.position.set(1.8, 0, 0);

  const poreMat = new THREE.MeshStandardMaterial({ color: 0x10b981, roughness: 0.35, emissive: 0x065f46 });
  const numHelices = 12;
  const barrelRadius = 0.75;
  const helixGeom = new THREE.CylinderGeometry(0.12, 0.12, 1.8, 8);

  for (let h = 0; h < numHelices; h++) {
    const angle = (h / numHelices) * Math.PI * 2;
    const helix = new THREE.Mesh(helixGeom, poreMat);
    helix.position.set(Math.cos(angle) * barrelRadius, 0, Math.sin(angle) * barrelRadius);
    glut.add(helix);
  }

  const plug = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 1.2, 16), new THREE.MeshStandardMaterial({
    color: 0x059669, transparent: true, opacity: 0.8
  }));
  glut.add(plug);

  registerBioObject(glut, 'GLUT4 Glucose Transporter Pore', 'Facilitated transport channel that translocates from intracellular storage vesicles into the membrane, allowing glucose influx.');
  glut.userData = { plug };
  return glut;
}

const glut4Channel = createGlut4Channel();
cellMembraneGroup.add(glut4Channel);
applyGlobalShadowProperties(cellMembraneGroup);

// ── GROUP 5: 3D NEURAL NETWORK GRAPH ────────────────────────────────────
const neuralGroup = new THREE.Group();
neuralGroup.position.set(0, -102, 0);
scene.add(neuralGroup);

function create3DNeuralNetwork() {
  const ann = new THREE.Group();
  const layerCounts = [8, 12, 8, 1];
  const layerX = [-6.5, -2.2, 2.2, 6.5];
  const nodeGeom = new THREE.SphereGeometry(0.24, 16, 16);

  const nodeMatInput  = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0284c7 });
  const nodeMatHidden = new THREE.MeshStandardMaterial({ color: 0x818cf8, emissive: 0x4f46e5 });
  const nodeMatOutput = new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xb91c1c });

  const allLayers = [];
  layerCounts.forEach((count, lIdx) => {
    const layerNodes = [];
    const mat = lIdx === 0 ? nodeMatInput : (lIdx === layerCounts.length - 1 ? nodeMatOutput : nodeMatHidden);
    const spacingY = 0.9;
    const startY = (count - 1) * spacingY * 0.5;

    for (let n = 0; n < count; n++) {
      const mesh = new THREE.Mesh(nodeGeom, mat);
      mesh.position.set(layerX[lIdx], startY - n * spacingY, (Math.random() - 0.5) * 0.4);
      ann.add(mesh);
      layerNodes.push(mesh);

      if (lIdx === 0 && MODEL_DATA.feature_names[n]) {
        registerBioObject(mesh, `Input Neuron: ${MODEL_DATA.feature_names[n]}`, `Clinical feature node feeding normalized patient measurement into hidden layer 1.`);
      } else if (lIdx === layerCounts.length - 1) {
        registerBioObject(mesh, 'Output Neuron (Risk Probability)', 'Sigmoid activation unit estimating binary diabetes risk probability in range [0.0, 1.0].');
      }
    }
    allLayers.push(layerNodes);
  });

  const synMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.18, blending: THREE.AdditiveBlending });
  for (let l = 0; l < allLayers.length - 1; l++) {
    allLayers[l].forEach(n1 => {
      allLayers[l + 1].forEach(n2 => {
        ann.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([n1.position, n2.position]), synMat));
      });
    });
  }

  const pulseCount = 45;
  const pulseGeom = new THREE.BufferGeometry();
  const pulsePos = new Float32Array(pulseCount * 3);
  const pulseData = [];

  for (let p = 0; p < pulseCount; p++) {
    const l = Math.floor(Math.random() * (allLayers.length - 1));
    pulseData.push({
      p1: allLayers[l][Math.floor(Math.random() * allLayers[l].length)].position,
      p2: allLayers[l + 1][Math.floor(Math.random() * allLayers[l + 1].length)].position,
      progress: Math.random(),
      speed: 0.015 + Math.random() * 0.02
    });
  }
  pulseGeom.setAttribute('position', new THREE.BufferAttribute(pulsePos, 3));
  const pulseParticles = new THREE.Points(pulseGeom, new THREE.PointsMaterial({
    color: 0xffffff, size: 0.16, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending
  }));
  ann.add(pulseParticles);

  ann.userData = { allLayers, pulseData, pulseParticles, pulsePos };
  return ann;
}

const neuralNetwork = create3DNeuralNetwork();
neuralGroup.add(neuralNetwork);

/* ==========================================================================
   4. MINIMAL BIOLOGICAL EXPLORATION: HOVER / TAP RAYCASTER
   ========================================================================== */
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2(-9999, -9999);
const bioTooltip = document.getElementById('bioTooltip');
const btName = document.getElementById('btName');
const btDesc = document.getElementById('btDesc');

let hoveredObject = null;
let lastCalculatedProb = 0.048; // Baseline default risk (~4.8%)

function onPointerMove(e) {
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

  if (bioTooltip && bioTooltip.classList.contains('visible')) {
    bioTooltip.style.left = `${Math.min(window.innerWidth - 180, Math.max(160, e.clientX))}px`;
    bioTooltip.style.top  = `${Math.max(80, e.clientY - 20)}px`;
  }
}

window.addEventListener('pointermove', onPointerMove, { passive: true });

function checkRaycast() {
  raycaster.setFromCamera(mouse, camera);
  // Cast against interactiveObjects AND all their deep children
  const intersects = raycaster.intersectObjects(interactiveObjects, true);

  for (let i = 0; i < intersects.length; i++) {
    // Walk up the parent chain until we find bioInfo or hit the root
    let target = intersects[i].object;
    let resolved = null;
    while (target) {
      if (target.userData && target.userData.bioInfo) {
        resolved = target;
        break;
      }
      target = target.parent;
    }

    if (resolved) {
      const info = resolved.userData.bioInfo;
      if (hoveredObject !== resolved) {
        hoveredObject = resolved;
        if (btName) btName.textContent = info.name;
        if (btDesc) btDesc.textContent = info.desc;
        const btDot = document.querySelector('.bt-dot');
        if (btDot) {
          if (info.name.includes('White Blood Cell') || info.name.includes('Leukocyte')) {
            btDot.style.background = '#f8fafc';
            btDot.style.boxShadow = '0 0 10px #ffffff';
          } else if (info.name.includes('Glucose')) {
            btDot.style.background = '#f59e0b';
            btDot.style.boxShadow = '0 0 10px #f59e0b';
          } else if (info.name.includes('Beta') || info.name.includes('Pancreas')) {
            btDot.style.background = '#10b981';
            btDot.style.boxShadow = '0 0 10px #10b981';
          } else {
            btDot.style.background = '#dc2626';
            btDot.style.boxShadow = '0 0 10px #dc2626';
          }
        }
        if (bioTooltip) bioTooltip.classList.add('visible');
        canvas.style.cursor = 'pointer';
      }
      return;
    }
  }

  if (hoveredObject) {
    hoveredObject = null;
    if (bioTooltip) bioTooltip.classList.remove('visible');
    canvas.style.cursor = 'default';
  }
}

window.addEventListener('pointerdown', (e) => {
  onPointerMove(e);
  checkRaycast();
});

/* ==========================================================================
   5. CONTINUOUS CINEMATIC SCROLL CAMERA CHOREOGRAPHY
   ========================================================================== */
const SCROLL_STAGES = [
  { trigger: '#hero',              cam: { x: 0,   y: 0,   z: 22 }, target: { x: 1.8, y: 0,   z: 0 } },
  { trigger: '#bloodstream',       cam: { x: 0,   y: -22, z: 7  }, target: { x: 0,   y: -22, z: -10 } },
  { trigger: '#pancreas',          cam: { x: 0,   y: -48, z: 9  }, target: { x: -0.8, y: -48, z: 2 } },
  { trigger: '#insulin',           cam: { x: 0.8, y: -75, z: 6  }, target: { x: 0,   y: -75, z: 0 } },
  { trigger: '#pathology',         cam: { x: 0,   y: -75, z: 12 }, target: { x: 0,   y: -75, z: 0 } },
  { trigger: '#prediction-portal', cam: { x: 0,   y: -102,z: 14 }, target: { x: 0,   y: -102,z: 0 } },
  { trigger: '#ai-research',       cam: { x: -2,  y: -102,z: 10 }, target: { x: 0,   y: -102,z: 0 } },
  { trigger: '#awareness',         cam: { x: 0,   y: -110,z: 20 }, target: { x: 0,   y: -102,z: 0 } }
];

const hudScale    = document.getElementById('hudScale');
const hudLocation = document.getElementById('hudLocation');
const hudChapter  = document.getElementById('hudChapter');
const hudTone     = document.getElementById('hudTone');

const targetCamPos    = new THREE.Vector3(SCROLL_STAGES[0].cam.x, SCROLL_STAGES[0].cam.y, SCROLL_STAGES[0].cam.z);
const targetCamLookAt = new THREE.Vector3(SCROLL_STAGES[0].target.x, SCROLL_STAGES[0].target.y, SCROLL_STAGES[0].target.z);

let stageMetrics = [];

function updateStageMetrics() {
  stageMetrics = SCROLL_STAGES.map(stage => {
    const el = document.querySelector(stage.trigger);
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const top = rect.top + scrollTop;
    const height = el.offsetHeight;
    return {
      trigger: stage.trigger,
      cam: stage.cam,
      target: stage.target,
      top: top,
      bottom: top + height,
      center: top + height * 0.5,
      el: el
    };
  }).filter(Boolean);
}

updateStageMetrics();
window.addEventListener('resize', updateStageMetrics);
window.addEventListener('load', updateStageMetrics);

let activeStageIndex = -1;

function updateContinuousCamera() {
  if (!stageMetrics.length) return;

  const scrollY = window.pageYOffset || document.documentElement.scrollTop;
  const viewportCenterY = scrollY + window.innerHeight * 0.5;

  const first = stageMetrics[0];
  const last  = stageMetrics[stageMetrics.length - 1];

  if (viewportCenterY <= first.center) {
    targetCamPos.set(first.cam.x, first.cam.y, first.cam.z);
    targetCamLookAt.set(first.target.x, first.target.y, first.target.z);
    setActiveStage(0);
    return;
  }

  if (viewportCenterY >= last.center) {
    targetCamPos.set(last.cam.x, last.cam.y, last.cam.z);
    targetCamLookAt.set(last.target.x, last.target.y, last.target.z);
    setActiveStage(stageMetrics.length - 1);
    return;
  }

  // Smooth S-Curve (Cosine) continuous interpolation between section midpoints
  for (let i = 0; i < stageMetrics.length - 1; i++) {
    const curr = stageMetrics[i];
    const next = stageMetrics[i + 1];

    if (viewportCenterY >= curr.center && viewportCenterY <= next.center) {
      const span = next.center - curr.center;
      let progress = span > 0 ? (viewportCenterY - curr.center) / span : 0;
      progress = Math.max(0, Math.min(1, progress));

      // Cosine easing has zero acceleration spike at endpoints (zero jerk)
      const easeT = 0.5 - 0.5 * Math.cos(progress * Math.PI);

      targetCamPos.x = THREE.MathUtils.lerp(curr.cam.x, next.cam.x, easeT);
      targetCamPos.y = THREE.MathUtils.lerp(curr.cam.y, next.cam.y, easeT);
      targetCamPos.z = THREE.MathUtils.lerp(curr.cam.z, next.cam.z, easeT);

      targetCamLookAt.x = THREE.MathUtils.lerp(curr.target.x, next.target.x, easeT);
      targetCamLookAt.y = THREE.MathUtils.lerp(curr.target.y, next.target.y, easeT);
      targetCamLookAt.z = THREE.MathUtils.lerp(curr.target.z, next.target.z, easeT);

      const closestStage = progress < 0.5 ? i : i + 1;
      setActiveStage(closestStage);
      break;
    }
  }
}

function setActiveStage(idx) {
  if (idx === activeStageIndex) return;
  activeStageIndex = idx;
  const stage = stageMetrics[idx];
  if (!stage) return;

  const el = stage.el;
  if (hudScale && el.dataset.scale) hudScale.textContent = el.dataset.scale;
  if (hudLocation && el.dataset.loc) hudLocation.textContent = el.dataset.loc;
  if (hudChapter && el.dataset.phase) hudChapter.textContent = el.dataset.phase;

  if (hudTone) {
    if (stage.trigger === '#hero') {
      hudTone.textContent = 'HOMEOSTATIC (NORMAL)';
      hudTone.className = 'hud-val hud-status-normal';
      hudTone.style.color = '';
    } else if (stage.trigger === '#bloodstream') {
      hudTone.textContent = 'LAMINAR FLOW (NORMAL)';
      hudTone.className = 'hud-val hud-status-normal';
      hudTone.style.color = '';
    } else if (stage.trigger === '#pancreas') {
      hudTone.textContent = 'ENDOCRINE BASAL';
      hudTone.className = 'hud-val hud-status-normal';
      hudTone.style.color = '';
    } else if (stage.trigger === '#insulin') {
      hudTone.textContent = 'RECEPTOR SENSITIVE';
      hudTone.className = 'hud-val hud-status-normal';
      hudTone.style.color = '';
    } else if (stage.trigger === '#pathology') {
      hudTone.textContent = 'COMPARATIVE STATES';
      hudTone.className = 'hud-val hud-status-normal';
      hudTone.style.color = '';
    } else if (stage.trigger === '#prediction-portal' || stage.trigger === '#ai-research') {
      const curScore = (lastCalculatedProb * 100).toFixed(1);
      if (lastCalculatedProb < 0.3) {
        hudTone.textContent = `MODEL RISK: ${curScore}% (LOW)`;
        hudTone.className = 'hud-val hud-status-normal';
        hudTone.style.color = '';
      } else if (lastCalculatedProb < 0.6) {
        hudTone.textContent = `MODEL RISK: ${curScore}% (BORDERLINE)`;
        hudTone.className = 'hud-val';
        hudTone.style.color = '#f59e0b';
      } else {
        hudTone.textContent = `MODEL RISK: ${curScore}% (HIGH)`;
        hudTone.className = 'hud-val';
        hudTone.style.color = '#ef4444';
      }
    }
  }

  document.querySelectorAll('.nav-links a').forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === stage.trigger);
  });

  // Focus class on active section
  document.querySelectorAll('.doc-chapter, .doc-hero').forEach(sec => {
    sec.classList.toggle('chapter-in-focus', sec === el);
  });
}

/* ==========================================================================
   6. MAIN RENDER LOOP & REAL-TIME BIOLOGICAL ANIMATIONS
   ========================================================================== */
let clock = new THREE.Clock();
let _frameCount = 0;

function animate() {
  requestAnimationFrame(animate);

  const delta = clock.getDelta();
  const time  = clock.getElapsedTime();

  // Throttle raycaster: run every 3 frames to save CPU/GPU
  if (_frameCount % 3 === 0) checkRaycast();
  _frameCount++;

  // 1. Full Body Gentle Micro-Rotation & Organ Dynamics
  if (completeHumanBody) {
    completeHumanBody.rotation.y = Math.sin(time * 0.2) * 0.14;
  }

  // 2. Cardiac Muscle Contraction (72 BPM rhythm: systole & diastole)
  if (heartMeshRef) {
    const heartBeat = 1.0 + 0.09 * Math.pow(Math.sin(time * 7.5), 6.0);
    heartMeshRef.scale.set(heartBeat, heartBeat, heartBeat);
  }

  // 3. Pulmonary Respiration (Gentle lung tidal expansion)
  if (leftLungRef && rightLungRef) {
    const breath = 1.0 + 0.04 * Math.sin(time * 1.8);
    leftLungRef.scale.set(breath, breath, breath);
    rightLungRef.scale.set(breath, breath, breath);
  }

  // 4. Floating Interstitial Particles
  if (floatingParticles) {
    floatingParticles.rotation.y = time * 0.02;
  }

  // 5. Bloodstream Fluid Flow: Laminar RBCs, Marginated WBC Rolling & Tumbling Glucose
  rbcList.forEach(rbc => {
    rbc.position.z += rbc.userData.vz;
    rbc.rotation.x += rbc.userData.rotX;
    rbc.rotation.y += rbc.userData.rotY;
    if (rbc.position.z > vesselLength * 0.45) rbc.position.z = -vesselLength * 0.45;
  });

  // Leukocyte margination & slow endothelial rolling with subtle selectin adhesion dynamics
  wbcList.forEach(wbc => {
    // Physiological selectin-integrin rolling hesitation
    const rollingHesitation = 0.85 + 0.25 * Math.sin(time * 1.2 + wbc.userData.phase);
    wbc.position.z += wbc.userData.vz * rollingHesitation;
    wbc.rotation.x += wbc.userData.rollX * rollingHesitation;
    wbc.rotation.y += wbc.userData.rollY * rollingHesitation;
    if (wbc.position.z > vesselLength * 0.45) wbc.position.z = -vesselLength * 0.45;
  });

  glucoseMolecules.forEach(g => {
    g.position.z += g.userData.vz;
    g.rotation.x += g.userData.rotX;
    if (g.position.z > vesselLength * 0.45) g.position.z = -vesselLength * 0.45;
  });

  // 6. Pancreas & Islet Exocytosis
  if (pancreasOrgan) pancreasOrgan.rotation.y = Math.sin(time * 0.3) * 0.06;
  if (isletCluster && isletCluster.userData.granules) {
    const pos = isletCluster.userData.granules.geometry.attributes.position.array;
    for (let i = 0; i < pos.length; i += 3) {
      pos[i + 2] += 0.015;
      if (pos[i + 2] > 2.0) pos[i + 2] = -1.0;
    }
    isletCluster.userData.granules.geometry.attributes.position.needsUpdate = true;
  }

  // 7. Lipid Bilayer Wave
  if (lipidBilayer && lipidBilayer.userData.heads) {
    lipidBilayer.userData.heads.forEach(h => {
      const dist = Math.sqrt(h.position.x * h.position.x + h.position.z * h.position.z);
      const wave = Math.sin(dist * 1.5 - time * 2.0) * 0.05;
      h.position.y = (h.position.y > 0 ? 0.45 : -0.45) + wave;
    });
  }

  // 8. 3D Neural Network Signal Propagation
  if (neuralNetwork && neuralNetwork.userData.pulseData) {
    const pData = neuralNetwork.userData.pulseData;
    const pPos  = neuralNetwork.userData.pulsePos;
    pData.forEach((p, idx) => {
      p.progress += p.speed;
      if (p.progress > 1) p.progress = 0;
      pPos[idx * 3]     = THREE.MathUtils.lerp(p.p1.x, p.p2.x, p.progress);
      pPos[idx * 3 + 1] = THREE.MathUtils.lerp(p.p1.y, p.p2.y, p.progress);
      pPos[idx * 3 + 2] = THREE.MathUtils.lerp(p.p1.z, p.p2.z, p.progress);
    });
    neuralNetwork.userData.pulseParticles.geometry.attributes.position.needsUpdate = true;
  }

  // Continuous camera scroll motion with Steadicam inertial smoothing
  updateContinuousCamera();

  const damping = PREFERS_REDUCED_MOTION ? 0.25 : 0.08;
  camera.position.x += (targetCamPos.x - camera.position.x) * damping;
  camera.position.y += (targetCamPos.y - camera.position.y) * damping;
  camera.position.z += (targetCamPos.z - camera.position.z) * damping;

  cameraTarget.x += (targetCamLookAt.x - cameraTarget.x) * damping;
  cameraTarget.y += (targetCamLookAt.y - cameraTarget.y) * damping;
  cameraTarget.z += (targetCamLookAt.z - cameraTarget.z) * damping;

  camera.lookAt(cameraTarget);
  renderer.render(scene, camera);
}

animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

/* ==========================================================================
   7. AUTHENTIC ANN PREDICTION & REAL SHAP XAI ALGORITHMS
   ========================================================================== */
function relu(arr) {
  return arr.map(v => Math.max(0, v));
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-Math.max(-25, Math.min(25, z))));
}

function forwardPassANN(zInput) {
  const W1 = MODEL_DATA.weights[0]; // 8 x 16
  const b1 = MODEL_DATA.biases[0];  // 16
  const W2 = MODEL_DATA.weights[1]; // 16 x 8
  const b2 = MODEL_DATA.biases[1];  // 8
  const W3 = MODEL_DATA.weights[2]; // 8 x 1
  const b3 = MODEL_DATA.biases[2];  // 1

  const h1 = new Array(16).fill(0);
  for (let j = 0; j < 16; j++) {
    let sum = b1[j];
    for (let i = 0; i < 8; i++) sum += zInput[i] * W1[i][j];
    h1[j] = Math.max(0, sum);
  }

  const h2 = new Array(8).fill(0);
  for (let k = 0; k < 8; k++) {
    let sum = b2[k];
    for (let j = 0; j < 16; j++) sum += h1[j] * W2[j][k];
    h2[k] = Math.max(0, sum);
  }

  let logit = b3[0];
  for (let k = 0; k < 8; k++) logit += h2[k] * W3[k][0];

  const prob = sigmoid(logit);
  return { prob, logit, h1, h2 };
}

function computeAuthenticSHAP(rawVector) {
  const cleanVector = rawVector.map((val, i) => {
    const fName = MODEL_DATA.feature_names[i];
    return (fName !== 'Pregnancies' && (isNaN(val) || val === null || val <= 0))
      ? MODEL_DATA.medians[fName]
      : (isNaN(val) ? 0 : val);
  });

  const zVector = cleanVector.map((val, i) => {
    return (val - MODEL_DATA.scaler_mean[i]) / MODEL_DATA.scaler_std[i];
  });

  const { prob: yPred } = forwardPassANN(zVector);
  const baselineExpected = MODEL_DATA.expected_value || 0.3547;
  const deltaTotal = yPred - baselineExpected;

  const steps = 6;
  const rawAttributions = new Array(8).fill(0);
  const eps = 1e-4;

  for (let s = 1; s <= steps; s++) {
    const alpha = s / steps;
    const zStep = zVector.map(z => z * alpha);
    const baseP = forwardPassANN(zStep).prob;

    for (let i = 0; i < 8; i++) {
      const zPerturbed = zStep.slice();
      zPerturbed[i] += eps;
      const pertP = forwardPassANN(zPerturbed).prob;
      const grad = (pertP - baseP) / eps;
      rawAttributions[i] += grad * zVector[i];
    }
  }

  for (let i = 0; i < 8; i++) rawAttributions[i] /= steps;

  const sumRaw = rawAttributions.reduce((a, b) => a + b, 0);
  const shapAttributions = rawAttributions.map(a => {
    return Math.abs(sumRaw) > 1e-6 ? (a / sumRaw) * deltaTotal : a;
  });

  return {
    rawVector: cleanVector,
    zVector,
    probability: yPred,
    baselineExpected,
    attributions: shapAttributions
  };
}

/* ==========================================================================
   8. CONNECT INPUTS TO 3D VISUALIZATION & PREDICTION PLATFORM
   ========================================================================== */
const predictionForm   = document.getElementById('predictionForm');
const riskScoreDisplay = document.getElementById('riskScoreDisplay');
const riskGaugeFill    = document.getElementById('riskGaugeFill');
const riskClassText    = document.getElementById('riskClassText');
const riskSubText      = document.getElementById('riskSubText');
const shapBarsList     = document.getElementById('shapBarsList');
const streamGlucVal    = document.getElementById('streamGlucVal');
const inputGlucose     = document.getElementById('inputGlucose');

function getFormFeatureVector() {
  const form = document.getElementById('predictionForm');
  if (!form) return [1, 100, 72, 23, 85, 26.5, 0.38, 33];

  return MODEL_DATA.feature_names.map(name => {
    const el = form.elements[name];
    const val = parseFloat(el ? el.value : '');
    return isNaN(val) ? MODEL_DATA.medians[name] : val;
  });
}

function runPredictionAndRender() {
  const vector = getFormFeatureVector();
  const res = computeAuthenticSHAP(vector);
  lastCalculatedProb = res.probability;

  const pct = (res.probability * 100).toFixed(1);
  if (riskScoreDisplay) riskScoreDisplay.textContent = `${pct}%`;
  if (riskGaugeFill) {
    riskGaugeFill.style.width = `${Math.min(100, res.probability * 100)}%`;
    if (res.probability < 0.30) {
      riskGaugeFill.style.backgroundColor = '#10b981';
      if (riskClassText) riskClassText.textContent = 'LOW ESTIMATED RISK';
      if (riskSubText) riskSubText.textContent = `(${pct}% — Below Population Baseline)`;
    } else if (res.probability < 0.60) {
      riskGaugeFill.style.backgroundColor = '#f59e0b';
      if (riskClassText) riskClassText.textContent = 'MODERATE / BORDERLINE RISK';
      if (riskSubText) riskSubText.textContent = `(${pct}% — Elevated Clinical Watch)`;
    } else {
      riskGaugeFill.style.backgroundColor = '#ef4444';
      if (riskClassText) riskClassText.textContent = 'HIGH ESTIMATED RISK';
      if (riskSubText) riskSubText.textContent = `(${pct}% — High Sensitivity)`;
    }
  }

  // Render Real SHAP Waterfall Bars
  if (shapBarsList) {
    shapBarsList.innerHTML = '';
    const featuresWithShap = MODEL_DATA.feature_names.map((name, i) => ({
      name,
      val: vector[i],
      unit: MODEL_DATA.feature_specs[name].unit,
      shap: res.attributions[i]
    })).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

    featuresWithShap.forEach(item => {
      const row = document.createElement('div');
      row.className = 'shap-row';
      const isPos = item.shap >= 0;
      const widthPct = Math.min(Math.abs(item.shap) * 150, 48);

      row.innerHTML = `
        <span class="shap-feature-name" title="${item.name}: ${item.val} ${item.unit}">
          ${item.name} (${item.val})
        </span>
        <div class="shap-track">
          <div class="shap-center-line"></div>
          ${isPos 
            ? `<div class="shap-fill-pos" style="width: ${widthPct}%;"></div>` 
            : `<div class="shap-fill-neg" style="width: ${widthPct}%;"></div>`
          }
        </div>
        <span class="shap-val-text ${isPos ? 'val-pos' : 'val-neg'}">
          ${isPos ? '+' : ''}${(item.shap * 100).toFixed(1)}%
        </span>
      `;
      shapBarsList.appendChild(row);
    });
  }

  // Update 3D Bloodstream and Neural Pulse speed
  const glucVal = vector[1];
  if (streamGlucVal) streamGlucVal.textContent = `Linked to Input (${glucVal} mg/dL)`;

  const activeCount = Math.floor(Math.min(1.0, Math.max(0.2, glucVal / 220)) * glucoseMolecules.length);
  glucoseMolecules.forEach((g, idx) => {
    g.visible = idx < activeCount;
    g.userData.vz = 0.05 + (glucVal / 250) * 0.12;
  });

  if (neuralNetwork && neuralNetwork.userData.pulseData) {
    const pulseFactor = 0.015 + (res.probability * 0.035);
    neuralNetwork.userData.pulseData.forEach(p => p.speed = pulseFactor * (0.8 + Math.random() * 0.4));
  }
}

if (inputGlucose) {
  inputGlucose.addEventListener('input', () => {
    const glucVal = parseFloat(inputGlucose.value) || 100;
    const activeCount = Math.floor(Math.min(1.0, Math.max(0.2, glucVal / 220)) * glucoseMolecules.length);
    glucoseMolecules.forEach((g, idx) => {
      g.visible = idx < activeCount;
      g.userData.vz = 0.05 + (glucVal / 250) * 0.12;
    });
    if (streamGlucVal) streamGlucVal.textContent = `Live Input: ${glucVal} mg/dL`;
  });
}

if (predictionForm) {
  predictionForm.addEventListener('submit', (e) => {
    e.preventDefault();
    runPredictionAndRender();
  });
}

const PRESETS = {
  healthy: {
    Pregnancies: 1, Glucose: 85, BloodPressure: 68, SkinThickness: 18,
    Insulin: 65, BMI: 21.8, DiabetesPedigreeFunction: 0.22, Age: 25
  },
  borderline: {
    Pregnancies: 2, Glucose: 122, BloodPressure: 78, SkinThickness: 28,
    Insulin: 135, BMI: 28.6, DiabetesPedigreeFunction: 0.45, Age: 44
  },
  highrisk: {
    Pregnancies: 4, Glucose: 178, BloodPressure: 88, SkinThickness: 38,
    Insulin: 210, BMI: 36.2, DiabetesPedigreeFunction: 0.88, Age: 52
  },
  median: MODEL_DATA.medians
};

document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const key = btn.dataset.preset;
    const data = PRESETS[key];
    if (!data || !predictionForm) return;

    Object.entries(data).forEach(([name, val]) => {
      if (predictionForm.elements[name]) predictionForm.elements[name].value = val;
    });
    runPredictionAndRender();
  });
});

const btnResetForm = document.getElementById('btnResetForm');
if (btnResetForm && predictionForm) {
  btnResetForm.addEventListener('click', () => {
    const defaults = {
      Pregnancies: 1, Glucose: 100, BloodPressure: 72, SkinThickness: 23,
      Insulin: 85, BMI: 26.5, DiabetesPedigreeFunction: 0.38, Age: 33
    };
    Object.entries(defaults).forEach(([name, val]) => {
      if (predictionForm.elements[name]) predictionForm.elements[name].value = val;
    });
    runPredictionAndRender();
  });
}

/* ==========================================================================
   9. POPULATE GLOBAL METRICS & IMPORTANCE
   ========================================================================== */
function populateGlobalMetrics() {
  const metricAuc  = document.getElementById('metricAuc');
  const metricAcc  = document.getElementById('metricAcc');
  const metricF1   = document.getElementById('metricF1');
  const metricBase = document.getElementById('metricBase');
  const globalList = document.getElementById('globalImportanceList');

  if (metricAuc)  metricAuc.textContent  = MODEL_DATA.metrics.roc_auc.toFixed(3);
  if (metricAcc)  metricAcc.textContent  = `${(MODEL_DATA.metrics.accuracy * 100).toFixed(1)}%`;
  if (metricF1)   metricF1.textContent   = MODEL_DATA.metrics.f1_score.toFixed(3);
  if (metricBase) metricBase.textContent = `${(MODEL_DATA.expected_value * 100).toFixed(1)}%`;

  if (globalList && MODEL_DATA.global_importance_pct) {
    globalList.innerHTML = '';
    const sorted = Object.entries(MODEL_DATA.global_importance_pct).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([name, pct]) => {
      const row = document.createElement('div');
      row.className = 'gi-row';
      row.innerHTML = `
        <span class="gi-name">${name}</span>
        <div class="gi-bar-track">
          <div class="gi-bar-fill" style="width: ${pct * 3.5}%;"></div>
        </div>
        <span class="gi-pct">${pct}%</span>
      `;
      globalList.appendChild(row);
    });
  }
}

populateGlobalMetrics();

/* ==========================================================================
   10. CHAPTER IV PATHOLOGY SWITCHER & OGTT CURVE
   ========================================================================== */
const ogttCanvas = document.getElementById('ogttCanvas');

function drawOGTTCurve(mode = 'normal') {
  if (!ogttCanvas) return;
  const ctx = ogttCanvas.getContext('2d');
  const w = ogttCanvas.width;
  const h = ogttCanvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  for (let y = 25; y < h - 25; y += 35) {
    ctx.beginPath();
    ctx.moveTo(35, y);
    ctx.lineTo(w - 15, y);
    ctx.stroke();
  }

  ctx.fillStyle = '#6b7280';
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.fillText('mg/dL', 5, 18);
  ctx.fillText('0 min', 35, h - 8);
  ctx.fillText('60 min', 200, h - 8);
  ctx.fillText('120 min', 380, h - 8);

  // Normal curve
  ctx.strokeStyle = mode === 'normal' ? '#10b981' : 'rgba(16, 185, 129, 0.25)';
  ctx.lineWidth = mode === 'normal' ? 3.5 : 1.5;
  ctx.beginPath();
  ctx.moveTo(40, 130);
  ctx.bezierCurveTo(110, 80, 160, 95, 230, 110);
  ctx.bezierCurveTo(300, 125, 370, 130, 420, 132);
  ctx.stroke();

  // Type 1 curve
  ctx.strokeStyle = mode === 'type1' ? '#ef4444' : 'rgba(239, 68, 68, 0.25)';
  ctx.lineWidth = mode === 'type1' ? 3.5 : 1.5;
  ctx.beginPath();
  ctx.moveTo(40, 110);
  ctx.bezierCurveTo(110, 50, 190, 25, 270, 22);
  ctx.bezierCurveTo(340, 20, 390, 18, 420, 16);
  ctx.stroke();

  // Type 2 curve
  ctx.strokeStyle = mode === 'type2' ? '#f59e0b' : 'rgba(245, 158, 11, 0.25)';
  ctx.lineWidth = mode === 'type2' ? 3.5 : 1.5;
  ctx.beginPath();
  ctx.moveTo(40, 105);
  ctx.bezierCurveTo(110, 65, 190, 55, 260, 60);
  ctx.bezierCurveTo(330, 68, 380, 75, 420, 80);
  ctx.stroke();
}

drawOGTTCurve('normal');

document.querySelectorAll('.pathology-mode-selector .mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.pathology-mode-selector .mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    drawOGTTCurve(btn.dataset.mode);
  });
});

/* ==========================================================================
   11. NAVBAR & INITIAL RUN
   ========================================================================== */
const navbar    = document.getElementById('navbar');
const navToggle = document.getElementById('navToggle');
const navLinks  = document.getElementById('navLinks');

window.addEventListener('scroll', () => {
  if (navbar) navbar.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });

if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => navLinks.classList.toggle('open'));
}

// Seamless, cinematic smooth navigation for all anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const href = this.getAttribute('href');
    if (!href || href === '#') return;
    const targetEl = document.querySelector(href);
    if (!targetEl) return;
    e.preventDefault();
    if (navLinks) navLinks.classList.remove('open');

    const rect = targetEl.getBoundingClientRect();
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    // Perfectly center target section in the viewport
    const targetY = rect.top + scrollTop - Math.max(0, (window.innerHeight - targetEl.offsetHeight) * 0.5);

    window.scrollTo({
      top: Math.max(0, targetY),
      behavior: 'smooth'
    });
  });
});

runPredictionAndRender();

console.log('INSIDE DIABETES: Engine initialized with full human body, internal organs, and authentic ANN.');
