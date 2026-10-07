const COMPONENT_MESSAGE = "isStreamlitMessage";
const paletteFamilies = {
  Brunette: ["Espresso", "Soft black", "Blue black", "Dark chocolate", "Mocha", "Chestnut", "Mahogany", "Burgundy", "Cherry cola", "Mushroom brown"],
  Warm: ["Copper", "Auburn", "Cinnamon", "Caramel", "Toffee", "Rose brown", "Fire red"],
  Blonde: ["Honey blonde", "Golden blonde", "Butter blonde", "Champagne blonde", "Ash blonde", "Silver", "Platinum"],
  Creative: ["Dusty rose", "Pastel pink", "Vivid magenta", "Violet", "Lavender", "Midnight blue", "Denim blue", "Teal", "Emerald"],
};
const diseaseGuidance = {
  "Alopecia Areata": ["Arrange a clinician or dermatologist review to confirm the cause of the hair loss.", "Avoid irritating the scalp and track when and where shedding appears.", "Do not start steroid or hair-growth medicines from an image result; discuss benefits and risks with a clinician."],
  "Contact Dermatitis": ["Pause a recently introduced hair or scalp product if it may be irritating you, and rinse the area gently.", "Choose fragrance-free, gentle products while the skin settles.", "Seek clinical advice for persistent, painful, blistering, or spreading irritation; do not apply medicated creams without guidance."],
  Folliculitis: ["Avoid picking, squeezing, or close shaving over irritated bumps; keep combs and clippers clean.", "Arrange medical advice if bumps are painful, spreading, recurring, or draining; seek urgent care for fever or rapidly worsening redness.", "A clinician can decide whether a medicated wash or other treatment is appropriate."],
  "Head Lice": ["Confirm live lice with a pharmacist or clinician before treating; image screening can be wrong.", "If confirmed, follow an approved treatment's label and ask a pharmacist which product is suitable.", "Use a fine-toothed comb and avoid sharing combs, hats, or hair accessories. Never use pet products, fuel, or household insect sprays."],
  "Lichen Planus": ["Arrange prompt dermatologist review because some scalp forms can scar and affect regrowth.", "Avoid scratching and pause products that sting or worsen irritation.", "Do not start steroid lotions or other medicated treatments without a clinician's direction."],
  "Male Pattern Baldness": ["A clinician or dermatologist can confirm the pattern and discuss evidence-based options and side effects.", "Take consistent photos over time and avoid unverified supplements or stopping prescribed medicines on your own.", "Use gentle hair care; shampoo or lotion cannot be selected reliably from this image score."],
  Psoriasis: ["Ask a clinician to confirm the cause, especially for thick scale, cracking, pain, or hair loss.", "Avoid picking scale and use gentle, fragrance-free hair care.", "A pharmacist or clinician can advise whether a medicated shampoo is suitable; follow its label and avoid combining treatments without advice."],
  "Seborrheic Dermatitis": ["If there is mild flaking, ask a pharmacist or clinician whether an over-the-counter anti-dandruff shampoo is suitable and follow its label.", "Avoid scratching and pause products that irritate the scalp.", "Get medical advice if symptoms are severe, persistent, painful, or not improving; this image cannot identify the right medicine."],
  "Telogen Effluvium": ["Arrange a clinician review to look for possible triggers such as recent illness, stress, or medication changes.", "Do not stop prescribed medicine or start iron, vitamins, or other supplements without medical advice.", "Use gentle handling and seek care if shedding is sudden, patchy, or accompanied by scalp symptoms."],
  "Tinea Capitis": ["Arrange prompt medical care for confirmation; scalp ringworm often needs prescription treatment, and shampoo alone may not clear it.", "Avoid sharing combs, hats, towels, or pillows, and clean hair tools.", "Do not use leftover antifungal or steroid medicines; ask a clinician or pharmacist what to do while awaiting care."],
};
const diseaseProducts = {
  "Alopecia Areata": [
    { name: "Gentle, fragrance-free shampoo", note: "For routine cleansing only; it does not treat the cause of patchy hair loss." },
    { name: "Soft wide-tooth comb", note: "Helps reduce pulling while detangling." },
  ],
  "Contact Dermatitis": [
    { name: "Fragrance-free, dye-free shampoo", note: "Choose a simple formula and stop products suspected of irritating the scalp." },
    { name: "Plain fragrance-free moisturizer", note: "For affected skin only if suitable; ask a pharmacist if the area is broken or weeping." },
  ],
  Folliculitis: [
    { name: "Mild, fragrance-free cleanser or shampoo", note: "Avoid scrubs and strongly fragranced products over irritated areas." },
    { name: "Clean combs and clipper guards", note: "Wash shared hair tools; do not use medicated washes unless a clinician recommends one." },
  ],
  "Head Lice": [
    { name: "Fine-toothed lice comb", note: "Use to check and remove lice or eggs according to local health guidance." },
    { name: "Pharmacy-approved lice treatment", note: "Only after live lice are confirmed; ask a pharmacist which option fits age, allergies, pregnancy, and local resistance guidance." },
  ],
  "Lichen Planus": [
    { name: "Gentle, fragrance-free shampoo", note: "Supportive cleansing only; avoid products that sting or increase irritation." },
    { name: "Dermatologist-recommended scalp treatment", note: "Prescription lotions may be used for confirmed disease; do not start steroid products from an image result." },
  ],
  "Male Pattern Baldness": [
    { name: "Gentle shampoo or cosmetic volumizing products", note: "May improve manageability or appearance but do not reverse hair loss." },
    { name: "Ask about over-the-counter topical minoxidil", note: "A clinician or pharmacist can check whether it is appropriate and explain use, risks, and when to avoid it." },
  ],
  Psoriasis: [
    { name: "Fragrance-free gentle shampoo", note: "Avoid picking scale or scrubbing inflamed skin." },
    { name: "Over-the-counter medicated shampoo", note: "Ask a pharmacist or clinician whether an ingredient such as salicylic acid or coal tar is suitable; follow the label and avoid broken skin." },
  ],
  "Seborrheic Dermatitis": [
    { name: "Over-the-counter anti-dandruff shampoo", note: "A pharmacist can help choose an appropriate active ingredient; follow its label and stop if irritation worsens." },
    { name: "Fragrance-free conditioner", note: "Use on hair lengths if needed; avoid adding new fragranced scalp products during a flare." },
  ],
  "Telogen Effluvium": [
    { name: "Gentle shampoo and wide-tooth comb", note: "Supportive care only; they do not address the underlying trigger." },
    { name: "Avoid unverified growth supplements", note: "Ask a clinician before starting vitamins or supplements; use them only when appropriate." },
  ],
  "Tinea Capitis": [
    { name: "Gentle shampoo and a clean comb", note: "Avoid sharing hair tools, hats, and towels while awaiting medical advice." },
    { name: "Ask about antifungal shampoo as an adjunct", note: "Scalp ringworm usually needs clinician-prescribed treatment; shampoo alone is not a cure." },
  ],
};

const elements = {
  upload: document.querySelector("#photo-input"),
  uploadLabel: document.querySelector("#upload-label"),
  uploadStatus: document.querySelector("#upload-status"),
  cameraStatus: document.querySelector("#camera-status"),
  uploadedPhoto: document.querySelector("#uploaded-photo"),
  cropDialog: document.querySelector("#crop-dialog"),
  cropImage: document.querySelector("#crop-image"),
  cropStage: document.querySelector("#crop-stage"),
  cropSelection: document.querySelector("#crop-selection"),
  cropStatus: document.querySelector("#crop-status"),
  cropReset: document.querySelector("#crop-reset"),
  cropApply: document.querySelector("#crop-apply"),
  cameraVideo: document.querySelector("#camera-video"),
  cameraControls: document.querySelector("#camera-controls"),
  startCamera: document.querySelector("#start-camera"),
  placeholder: document.querySelector("#photo-placeholder"),
  scanLine: document.querySelector("#scan-line"),
  results: document.querySelector("#analysis"),
  profileContent: document.querySelector("#profile-content"),
  error: document.querySelector("#analysis-error"),
  conditionAssessment: document.querySelector("#condition-assessment"),
  cuts: document.querySelector("#cuts"),
  haircutPreview: document.querySelector("#haircut-preview"),
  mapping: document.querySelector("#hair-mapping"),
  diagnostics: document.querySelector("#diagnostics"),
  colorStudio: document.querySelector("#color-studio"),
  type: document.querySelector("#hair-type"),
  confidence: document.querySelector("#type-confidence"),
  area: document.querySelector("#hair-area"),
  confidenceFill: document.querySelector("#confidence-fill"),
  haircutGrid: document.querySelector("#haircut-grid"),
  styleDirections: document.querySelector("#style-directions"),
  lengthDirections: document.querySelector("#length-directions"),
  visibleLength: document.querySelector("#visible-hair-length"),
  colorRegionButtons: [...document.querySelectorAll("[data-color-region]")],
  colorRegionLabel: document.querySelector("#color-region-label"),
  comparisonStage: document.querySelector("#comparison-stage"),
  afterClip: document.querySelector("#after-clip"),
  comparisonDivider: document.querySelector("#comparison-divider"),
  comparisonSlider: document.querySelector("#comparison-slider"),
  families: document.querySelector("#color-families"),
  palette: document.querySelector("#color-palette"),
  selectedColorName: document.querySelector("#selected-color-name"),
  blend: document.querySelector("#blend-slider"),
  blendValue: document.querySelector("#blend-value"),
  beforeCanvas: document.querySelector("#before-canvas"),
  afterCanvas: document.querySelector("#after-canvas"),
  mapCanvas: document.querySelector("#hair-map-canvas"),
  cutBaseCanvas: document.querySelector("#cut-base-canvas"),
  cutTryonCanvas: document.querySelector("#cut-tryon-canvas"),
  cutReferenceImage: document.querySelector("#cut-reference-image"),
  tryonBlend: document.querySelector("#tryon-blend-slider"),
  tryonBlendValue: document.querySelector("#tryon-blend-value"),
  download: document.querySelector("#download-preview"),
  selectionLabel: document.querySelector("#selection-label"),
};

const beforeContext = elements.beforeCanvas.getContext("2d", { willReadFrequently: true });
const afterContext = elements.afterCanvas.getContext("2d", { willReadFrequently: true });
const mapContext = elements.mapCanvas.getContext("2d", { willReadFrequently: true });
const cutBaseContext = elements.cutBaseCanvas.getContext("2d");
const cutTryonContext = elements.cutTryonCanvas.getContext("2d");
let currentResult = null;
let originalImage = null;
let maskImage = null;
let activeHaircut = null;
let cameraStream = null;
let selection = null;
let pendingPhoto = null;
let cropStart = null;
let cropSelection = null;
let activeFamily = "Brunette";
let activePresentation = "Feminine";
let activeLength = "All lengths";
let activeColorRegion = "whole";
let hairMaskBounds = null;
let selectedColor = { name: "Espresso", rgb: [55, 35, 28] };

function sendToStreamlit(type, data = {}) {
  window.parent.postMessage({ [COMPONENT_MESSAGE]: true, type, ...data }, "*");
}

function setFrameHeight() {
  sendToStreamlit("streamlit:setFrameHeight", {
    height: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
  });
}

const frameResizeObserver = new ResizeObserver(() => setFrameHeight());
frameResizeObserver.observe(document.body);
frameResizeObserver.observe(document.documentElement);
window.addEventListener("resize", setFrameHeight);
window.addEventListener("resize", sizeComparisonStage);

function sizeComparisonStage() {
  const width = elements.beforeCanvas.width;
  const height = elements.beforeCanvas.height;
  const availableWidth = elements.comparisonStage.parentElement.clientWidth;
  if (!width || !height || !availableWidth) return;
  const scale = Math.min(availableWidth / width, 340 / height);
  elements.comparisonStage.style.width = `${width * scale}px`;
  elements.comparisonStage.style.height = `${height * scale}px`;
}

function updateComparisonWipe() {
  const position = Number(elements.comparisonSlider.value);
  elements.comparisonStage.style.setProperty("--comparison-position", `${position}%`);
  elements.afterClip.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
  elements.comparisonDivider.style.left = `${position}%`;
}

elements.comparisonSlider.addEventListener("input", updateComparisonWipe);
updateComparisonWipe();

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function showUploadState(imageData, filename) {
  elements.uploadedPhoto.src = imageData;
  elements.uploadedPhoto.hidden = false;
  elements.placeholder.hidden = true;
  elements.uploadLabel.textContent = "Choose another photo";
  elements.uploadStatus.textContent = filename ? `Ready to analyze: ${filename}` : "Photo ready to analyze.";
}

function submitImage(imageData, filename) {
  showUploadState(imageData, filename);
  elements.scanLine.hidden = false;
  sendToStreamlit("streamlit:setComponentValue", {
    value: { imageData, filename },
    dataType: "json",
  });
}

function updateCropSelection(point) {
  if (!cropStart || !point) {
    elements.cropSelection.hidden = true;
    elements.cropApply.disabled = true;
    return;
  }

  const left = Math.min(cropStart.x, point.x);
  const top = Math.min(cropStart.y, point.y);
  const width = Math.abs(point.x - cropStart.x);
  const height = Math.abs(point.y - cropStart.y);
  cropSelection = { left, top, width, height };
  elements.cropSelection.hidden = false;
  elements.cropSelection.style.left = `${left}px`;
  elements.cropSelection.style.top = `${top}px`;
  elements.cropSelection.style.width = `${width}px`;
  elements.cropSelection.style.height = `${height}px`;
  elements.cropApply.disabled = width < 24 || height < 24;
  elements.cropReset.disabled = false;
  elements.cropStatus.textContent = elements.cropApply.disabled
    ? "Drag to select a larger area."
    : `Selected area: ${Math.round(width)} × ${Math.round(height)} pixels.`;
}

function resetCropSelection() {
  cropStart = null;
  cropSelection = null;
  elements.cropSelection.hidden = true;
  elements.cropReset.disabled = true;
  elements.cropApply.disabled = true;
  elements.cropStatus.textContent = "No crop area selected.";
}

async function openCropDialog(imageData, filename) {
  pendingPhoto = { imageData, filename };
  elements.cropImage.src = imageData;

  if (!elements.cropImage.complete || !elements.cropImage.naturalWidth) {
    await new Promise((resolve, reject) => {
      elements.cropImage.addEventListener("load", resolve, { once: true });
      elements.cropImage.addEventListener(
        "error",
        () => reject(new Error("Could not display that image.")),
        { once: true },
      );
    });
  }

  resetCropSelection();
  elements.cropDialog.showModal();
}

function cropPoint(event) {
  const bounds = elements.cropStage.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(bounds.width, event.clientX - bounds.left)),
    y: Math.max(0, Math.min(bounds.height, event.clientY - bounds.top)),
  };
}

elements.cropStage.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  event.preventDefault();
  cropStart = cropPoint(event);
  elements.cropStage.setPointerCapture(event.pointerId);
  updateCropSelection(cropStart);
});
elements.cropStage.addEventListener("pointermove", (event) => {
  if (cropStart) updateCropSelection(cropPoint(event));
});
elements.cropStage.addEventListener("pointerup", (event) => {
  if (cropStart) updateCropSelection(cropPoint(event));
  cropStart = null;
});
elements.cropStage.addEventListener("pointercancel", () => {
  cropStart = null;
});
elements.cropReset.addEventListener("click", resetCropSelection);

function closeCropDialog() {
  elements.cropDialog.close();
}

document.querySelector("#crop-cancel").addEventListener("click", closeCropDialog);
document.querySelector("#crop-close").addEventListener("click", closeCropDialog);
elements.cropDialog.addEventListener("close", () => {
  if (pendingPhoto) {
    elements.uploadStatus.textContent = "Cropping cancelled. Your previous photo is unchanged.";
  }
  pendingPhoto = null;
  elements.cropImage.removeAttribute("src");
  elements.uploadLabel.textContent = elements.uploadedPhoto.hidden
    ? "Choose a photo"
    : "Choose another photo";
});

elements.cropApply.addEventListener("click", () => {
  if (!pendingPhoto || !cropSelection) return;

  const scaleX = elements.cropImage.naturalWidth / elements.cropStage.clientWidth;
  const scaleY = elements.cropImage.naturalHeight / elements.cropStage.clientHeight;
  const sourceX = Math.round(cropSelection.left * scaleX);
  const sourceY = Math.round(cropSelection.top * scaleY);
  const sourceWidth = Math.min(
    elements.cropImage.naturalWidth - sourceX,
    Math.round(cropSelection.width * scaleX),
  );
  const sourceHeight = Math.min(
    elements.cropImage.naturalHeight - sourceY,
    Math.round(cropSelection.height * scaleY),
  );
  if (sourceWidth < 1 || sourceHeight < 1) {
    elements.cropStatus.textContent = "Select a valid area of the photo and try again.";
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = sourceWidth;
  canvas.height = sourceHeight;
  const context = canvas.getContext("2d");
  if (!context) {
    elements.cropStatus.textContent = "Photo cropping is unavailable in this browser.";
    return;
  }
  context.drawImage(
    elements.cropImage,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    0,
    0,
    canvas.width,
    canvas.height,
  );

  const croppedPhoto = canvas.toDataURL("image/jpeg", 0.88);
  const { filename } = pendingPhoto;
  pendingPhoto = null;
  closeCropDialog();
  submitImage(croppedPhoto, filename);
});

function stopCamera() {
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  elements.cameraVideo.srcObject = null;
  elements.cameraVideo.hidden = true;
  elements.cameraControls.hidden = true;
  elements.startCamera.hidden = false;
  elements.uploadedPhoto.hidden = !elements.uploadedPhoto.src;
  elements.placeholder.hidden = Boolean(elements.uploadedPhoto.src);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    elements.cameraStatus.textContent = "Live camera is unavailable in this browser. Choose a photo instead.";
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "user" } },
    });
    elements.cameraVideo.srcObject = cameraStream;
    elements.cameraVideo.hidden = false;
    elements.uploadedPhoto.hidden = true;
    elements.placeholder.hidden = true;
    elements.cameraControls.hidden = false;
    elements.startCamera.hidden = true;
    elements.cameraStatus.textContent = "Camera ready. Capture when you are in frame.";
    await elements.cameraVideo.play();
  } catch {
    stopCamera();
    elements.cameraStatus.textContent = "Camera permission was denied or no camera was found. Choose a photo instead.";
  }
}

elements.startCamera.addEventListener("click", startCamera);
document.querySelector("#stop-camera").addEventListener("click", () => {
  stopCamera();
  elements.cameraStatus.textContent = "Camera closed.";
});

document.querySelector("#capture-camera").addEventListener("click", () => {
  const video = elements.cameraVideo;
  if (!cameraStream || !video.videoWidth) return;
  const scale = Math.min(1, 1500 / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const context = canvas.getContext("2d");
  if (!context) {
    elements.cameraStatus.textContent = "Photo capture is unavailable in this browser.";
    return;
  }
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL("image/jpeg", 0.88);
  stopCamera();
  elements.cameraStatus.textContent = "Camera photo captured. Crop it before analysis.";
  openCropDialog(imageData, "Live camera photo").catch((error) => {
    pendingPhoto = null;
    elements.cameraStatus.textContent = error.message;
  });
});

window.addEventListener("beforeunload", () => {
  cameraStream?.getTracks().forEach((track) => track.stop());
});

async function compressImage(file) {
  const image = new Image();
  const fileData = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read that image."));
    reader.readAsDataURL(file);
  });
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error("Could not decode that image."));
    image.src = fileData;
    if (image.complete && image.naturalWidth > 0) resolve();
  });

  const maxDimension = 1500;
  const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
  canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.88);
}

elements.upload.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  elements.uploadStatus.textContent = "Preparing your photo...";
  elements.uploadLabel.textContent = "Preparing photo";
  stopCamera();
  try {
    const imageData = await compressImage(file);
    await openCropDialog(imageData, file.name);
  } catch (error) {
    pendingPhoto = null;
    elements.uploadStatus.textContent = error instanceof Error
      ? error.message
      : "Could not prepare that image.";
    elements.uploadLabel.textContent = elements.uploadedPhoto.hidden
      ? "Choose a photo"
      : "Choose another photo";
  } finally {
    elements.upload.value = "";
  }
});

document.querySelector("#change-photo").addEventListener("click", () => {
  elements.upload.click();
  document.querySelector("#upload-stage").scrollIntoView({ behavior: "smooth", block: "center" });
});
document.querySelector("#close-tryon").addEventListener("click", () => {
  elements.haircutPreview.hidden = true;
});

function buildStyleDirections(recommendations) {
  const presentations = new Set(recommendations.map((item) => item.presentation));
  const directions = ["All styles", "Feminine", "Masculine"].filter(
    (direction) => direction === "All styles" || presentations.has(direction),
  );
  if (!directions.includes(activePresentation)) activePresentation = "All styles";
  elements.styleDirections.replaceChildren();
  directions.forEach((direction) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "style-direction-button";
    button.textContent = {
      "All styles": "All styles",
      Feminine: "Women's styles",
      Masculine: "Men's styles",
    }[direction];
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(activePresentation === direction));
    button.addEventListener("click", () => {
      activePresentation = direction;
      buildStyleDirections(recommendations);
      addHaircutCards(recommendations);
      elements.haircutPreview.hidden = true;
    });
    elements.styleDirections.append(button);
  });
}

function buildLengthDirections(recommendations, detectedLength) {
  const availableLengths = new Set(recommendations.map((item) => item.length));
  const lengths = ["All lengths", "short", "medium", "long"].filter(
    (length) => length === "All lengths" || availableLengths.has(length),
  );
  if (activeLength === "All lengths" && availableLengths.has(detectedLength)) {
    activeLength = detectedLength;
  }
  if (!lengths.includes(activeLength)) activeLength = "All lengths";
  elements.visibleLength.textContent = detectedLength && detectedLength !== "unclear"
    ? `Photo estimate: ${detectedLength}`
    : "Photo estimate: unclear";
  elements.lengthDirections.replaceChildren();
  lengths.forEach((length) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "style-direction-button";
    button.textContent = length === "All lengths" ? "All" : length[0].toUpperCase() + length.slice(1);
    button.setAttribute("aria-pressed", String(activeLength === length));
    button.addEventListener("click", () => {
      activeLength = length;
      buildLengthDirections(recommendations, detectedLength);
      addHaircutCards(recommendations);
      elements.haircutPreview.hidden = true;
    });
    elements.lengthDirections.append(button);
  });
}

function addHaircutCards(recommendations) {
  elements.haircutGrid.replaceChildren();
  const visibleRecommendations = recommendations.filter(
    (recommendation) => (activePresentation === "All styles"
      || recommendation.presentation === activePresentation)
      && (activeLength === "All lengths" || recommendation.length === activeLength),
  );
  visibleRecommendations.forEach((recommendation, index) => {
    const card = document.createElement("article");
    card.className = "haircut-card";

    const imageWrap = document.createElement("div");
    imageWrap.className = "cut-image-wrap";
    const referenceUnavailable = document.createElement("div");
    referenceUnavailable.className = "cut-image-placeholder";
    referenceUnavailable.textContent = "Reference photo unavailable in this deployment.";
    const image = document.createElement("img");
    image.className = "cut-image";
    image.alt = `${recommendation.name} reference hairstyle`;
    image.loading = "lazy";
    if (recommendation.preview) {
      image.addEventListener("error", () => {
        image.remove();
        imageWrap.prepend(referenceUnavailable);
      }, { once: true });
      image.src = recommendation.preview;
      imageWrap.append(image);
    } else {
      imageWrap.append(referenceUnavailable);
    }
    const number = document.createElement("span");
    number.className = "cut-number";
    number.textContent = String(index + 1).padStart(2, "0");
    imageWrap.append(number);

    const info = document.createElement("div");
    info.className = "cut-info";
    const meta = document.createElement("span");
    meta.className = "cut-meta";
    meta.textContent = `${recommendation.presentation || "STYLE"} STYLE REFERENCE`;
    const title = document.createElement("h3");
    title.textContent = recommendation.name;
    const description = document.createElement("p");
    description.textContent = recommendation.description;
    const detail = document.createElement("span");
    detail.className = "cut-detail";
    detail.textContent = recommendation.detail;
    const preview = document.createElement("button");
    preview.type = "button";
    preview.className = "cut-preview-button";
    preview.textContent = recommendation.hairData
      ? "Preview on my photo"
      : "Preview unavailable";
    preview.disabled = !recommendation.hairData;
    if (recommendation.hairData) {
      preview.addEventListener("click", () => showHaircutPreview(recommendation));
    } else {
      preview.title = "Reference hairstyle images are not included in this deployment.";
    }
    info.append(meta, title, description, detail, preview);
    card.append(imageWrap, info);
    elements.haircutGrid.append(card);
  });
}

function renderProfile(result) {
  elements.results.hidden = false;
  elements.scanLine.hidden = true;
  elements.error.hidden = true;
  elements.profileContent.hidden = true;
  elements.conditionAssessment.hidden = true;
  elements.cuts.hidden = true;
  elements.haircutPreview.hidden = true;
  elements.mapping.hidden = true;
  elements.diagnostics.hidden = true;
  elements.colorStudio.hidden = true;
  elements.uploadStatus.textContent = "Analysis complete.";

  if (result.error) {
    elements.error.textContent = result.error;
    elements.error.hidden = false;
    setFrameHeight();
    return;
  }

  renderDiagnostics(result);
  elements.diagnostics.hidden = false;
  renderConditionAssessment(result);

  if (!result.hairDetected) {
    const nonHairConfidence = formatPercent(
      result.nonHairConfidence ?? result.presenceConfidence,
    );
    elements.error.textContent = `The hair-presence model labeled this image as non-hair (${nonHairConfidence} confidence). Hair-type and color analysis are skipped; the separate condition screen is still shown below.`;
    elements.error.hidden = false;
    setFrameHeight();
    return;
  }

  currentResult = result;
  elements.type.textContent = result.hairType;
  elements.confidence.textContent = formatPercent(result.typeConfidence);
  elements.area.textContent = `${result.hairArea.toFixed(1)}%`;
  document.querySelector("#mapping-area").textContent = `${result.hairArea.toFixed(1)}%`;
  document.querySelector("#image-dimensions").textContent = `${result.imageWidth} × ${result.imageHeight}`;
  elements.confidenceFill.style.width = `${Math.max(0, Math.min(100, result.typeConfidence * 100))}%`;
  elements.profileContent.hidden = false;
  elements.cuts.hidden = false;
  elements.mapping.hidden = false;
  elements.colorStudio.hidden = false;
  buildStyleDirections(result.recommendations || []);
  activeLength = result.visibleHairLength || "All lengths";
  buildLengthDirections(result.recommendations || [], activeLength);
  addHaircutCards(result.recommendations || []);
  drawCanvases(result);
  buildColorControls(window.componentArgs.colors || {});
  document.querySelector("#upload-stage").classList.add("has-result");
  setFrameHeight();
}

function renderConditionAssessment(result) {
  if (!result.conditionEvaluated) return;
  const isDisease = result.conditionLabel === "Disease";
  elements.conditionAssessment.hidden = false;
  elements.conditionAssessment.classList.toggle("is-disease", isDisease);
  document.querySelector("#condition-headline").textContent = isDisease
    ? "A possible condition pattern was flagged."
    : "No visible condition flagged.";
  document.querySelector("#condition-badge").textContent = isDisease
    ? "REVIEW RESULT"
    : "NO CONDITION FLAGGED";
  document.querySelector("#condition-gate-label").textContent = isDisease
    ? "Disease"
    : "Normal / no visible condition";
  document.querySelector("#condition-gate-confidence").textContent = formatPercent(
    result.conditionConfidence,
  );
  document.querySelector("#condition-description").textContent = result.hairDetected
    ? isDisease
      ? "The condition gate flagged this image. The separate disease classifier's highest-scoring class is shown as a possibility below."
      : "The condition gate did not flag a visible condition in this image."
    : "The condition screen ran, but the separate hair-presence model did not recognize this as a hair image. A scalp-focused photo may produce a more relevant result.";

  const classResult = document.querySelector("#condition-class-result");
  const probabilityDetails = document.querySelector("#condition-probability-details");
  const guidance = document.querySelector("#condition-guidance");
  classResult.hidden = !isDisease || !result.diseaseLabel;
  probabilityDetails.hidden = !isDisease || !result.diseaseScores?.length;
  guidance.hidden = !isDisease;
  if (isDisease) {
    const advice = diseaseGuidance[result.diseaseLabel] || [
      "Arrange review with a qualified clinician to confirm what is causing the visible changes.",
      "Avoid starting prescription or medicated products from this image score; ask a clinician or pharmacist what is appropriate.",
      "Seek prompt care for pain, fever, rapid worsening, or sudden hair loss.",
    ];
    document.querySelector("#condition-guidance-summary").textContent = result.diseaseLabel
      ? `General precautions for a possible ${result.diseaseLabel} pattern. This is not a confirmed diagnosis.`
      : "The image screen raised a possible concern, but no class could be identified. A clinician can examine the scalp and recommend appropriate care.";
    const steps = document.querySelector("#condition-guidance-steps");
    steps.replaceChildren();
    advice.forEach((text) => {
      const step = document.createElement("li");
      step.textContent = text;
      steps.append(step);
    });
    const productList = document.querySelector("#condition-products-list");
    const products = diseaseProducts[result.diseaseLabel] || [
      { name: "Gentle, fragrance-free shampoo", note: "A supportive option only; it does not treat an unknown scalp condition." },
      { name: "Clinician- or pharmacist-recommended product", note: "Confirm the cause before using medicated lotions, shampoos, or treatments." },
    ];
    productList.replaceChildren();
    products.forEach(({ name, note }) => {
      const item = document.createElement("li");
      const productName = document.createElement("strong");
      productName.textContent = name;
      const productNote = document.createElement("span");
      productNote.textContent = note;
      item.append(productName, productNote);
      productList.append(item);
    });
  }
  if (isDisease && result.diseaseLabel) {
    document.querySelector("#condition-class-label").textContent = result.diseaseLabel;
    document.querySelector("#condition-class-confidence").textContent = formatPercent(
      result.diseaseConfidence,
    );
    const rows = document.querySelector("#disease-probabilities");
    rows.replaceChildren();
    result.diseaseClasses.forEach((name, index) => {
      const row = document.createElement("div");
      row.className = "probability-row";
      const label = document.createElement("span");
      label.textContent = name;
      const track = document.createElement("span");
      track.className = "probability-track";
      const fill = document.createElement("span");
      fill.style.width = `${result.diseaseScores[index] * 100}%`;
      track.append(fill);
      const value = document.createElement("strong");
      value.textContent = formatPercent(result.diseaseScores[index]);
      row.append(label, track, value);
      rows.append(row);
    });
  }
}

function renderDiagnostics(result) {
  const hairScore = result.hairConfidence ?? 0;
  const nonHairScore = result.nonHairConfidence ?? 0;
  document.querySelector("#diag-hair-score").textContent = formatPercent(hairScore);
  document.querySelector("#diag-nonhair-score").textContent = formatPercent(nonHairScore);
  document.querySelector("#diag-hair-fill").style.width = `${hairScore * 100}%`;
  document.querySelector("#diag-nonhair-fill").style.width = `${nonHairScore * 100}%`;
  document.querySelector("#diag-type-score").textContent = result.typeConfidence === undefined
    ? "Skipped"
    : formatPercent(result.typeConfidence);
  document.querySelector("#diag-type-fill").style.width = `${(result.typeConfidence || 0) * 100}%`;
  document.querySelector("#diag-hair-area").textContent = result.hairArea === undefined
    ? "Skipped"
    : `${result.hairArea.toFixed(1)}%`;
  document.querySelector("#diag-area-fill").style.width = `${result.hairArea || 0}%`;

  const probabilityList = document.querySelector("#type-probabilities");
  probabilityList.replaceChildren();
  (result.typeClasses || []).forEach((name, index) => {
    const row = document.createElement("div");
    row.className = "probability-row";
    const label = document.createElement("span");
    label.textContent = name;
    const track = document.createElement("span");
    track.className = "probability-track";
    const fill = document.createElement("span");
    fill.style.width = `${(result.typeScores[index] || 0) * 100}%`;
    track.append(fill);
    const value = document.createElement("strong");
    value.textContent = formatPercent(result.typeScores[index] || 0);
    row.append(label, track, value);
    probabilityList.append(row);
  });
  if (!result.typeClasses?.length) {
    const note = document.createElement("p");
    note.className = "probability-empty";
    note.textContent = "Type classification is skipped when the hair-presence gate rejects the image.";
    probabilityList.append(note);
  }
}

function renderHairMap() {
  if (!originalImage?.complete || !maskImage?.complete) return;
  const width = originalImage.naturalWidth;
  const height = originalImage.naturalHeight;
  elements.mapCanvas.width = width;
  elements.mapCanvas.height = height;
  mapContext.drawImage(originalImage, 0, 0);

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
  maskContext.drawImage(maskImage, 0, 0, width, height);
  const mask = maskContext.getImageData(0, 0, width, height).data;
  const output = mapContext.getImageData(0, 0, width, height);
  for (let index = 0; index < mask.length; index += 4) {
    if (mask[index] < 128) continue;
    const pixel = index / 4;
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    const edge = x === 0 || y === 0 || x === width - 1 || y === height - 1
      || mask[index - 4] < 128
      || mask[index + 4] < 128
      || mask[index - width * 4] < 128
      || mask[index + width * 4] < 128;
    const tint = edge ? [226, 139, 106] : [49, 129, 104];
    const strength = edge ? 0.48 : 0.18;
    for (let channel = 0; channel < 3; channel += 1) {
      output.data[index + channel] = Math.round(
        output.data[index + channel] * (1 - strength) + tint[channel] * strength,
      );
    }
  }
  mapContext.putImageData(output, 0, 0);
}

function renderHaircutPreview(recommendation) {
  if (!currentResult || !originalImage?.complete || !maskImage?.complete) return;
  activeHaircut = recommendation;
  const width = originalImage.naturalWidth;
  const height = originalImage.naturalHeight;
  elements.cutBaseCanvas.width = width;
  elements.cutBaseCanvas.height = height;
  elements.cutTryonCanvas.width = width;
  elements.cutTryonCanvas.height = height;
  cutBaseContext.drawImage(originalImage, 0, 0);
  cutTryonContext.drawImage(originalImage, 0, 0);
  document.querySelector("#tryon-cut-name").textContent = recommendation.name;
  document.querySelector("#tryon-cut-description").textContent = recommendation.description;
  elements.cutReferenceImage.src = recommendation.preview;
  elements.cutReferenceImage.alt = `${recommendation.name} reference photo`;
  elements.haircutPreview.hidden = false;

  if (!recommendation.hairData) {
    const reference = new Image();
    reference.onload = () => cutTryonContext.drawImage(reference, 0, 0, width, height);
    reference.src = recommendation.preview;
    return;
  }

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
  maskContext.drawImage(maskImage, 0, 0, width, height);
  const pixels = maskContext.getImageData(0, 0, width, height).data;
  let left = width;
  let top = height;
  let right = 0;
  let bottom = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (pixels[(y * width + x) * 4] < 128) continue;
      left = Math.min(left, x);
      top = Math.min(top, y);
      right = Math.max(right, x);
      bottom = Math.max(bottom, y);
    }
  }
  if (right <= left || bottom <= top) return;

  const haircutImage = new Image();
  haircutImage.onload = () => {
    const overlayCanvas = document.createElement("canvas");
    overlayCanvas.width = width;
    overlayCanvas.height = height;
    const overlayContext = overlayCanvas.getContext("2d");
    const targetWidth = right - left;
    const targetHeight = bottom - top;
    const scale = Math.min(
      targetWidth / haircutImage.naturalWidth,
      targetHeight / haircutImage.naturalHeight,
    );
    const overlayWidth = haircutImage.naturalWidth * scale;
    const overlayHeight = haircutImage.naturalHeight * scale;
    overlayContext.drawImage(
      haircutImage,
      left + (targetWidth - overlayWidth) / 2,
      top,
      overlayWidth,
      overlayHeight,
    );
    const targetMask = maskContext.getImageData(0, 0, width, height);
    for (let index = 0; index < targetMask.data.length; index += 4) {
      targetMask.data[index + 3] = targetMask.data[index];
      targetMask.data[index] = 255;
      targetMask.data[index + 1] = 255;
      targetMask.data[index + 2] = 255;
    }
    maskContext.putImageData(targetMask, 0, 0);
    overlayContext.globalCompositeOperation = "destination-in";
    overlayContext.drawImage(maskCanvas, 0, 0);
    overlayContext.globalCompositeOperation = "source-over";
    cutTryonContext.globalAlpha = Number(elements.tryonBlend.value) / 100;
    cutTryonContext.drawImage(overlayCanvas, 0, 0);
    cutTryonContext.globalAlpha = 1;
  };
  haircutImage.src = recommendation.hairData;
}

function showHaircutPreview(recommendation) {
  renderHaircutPreview(recommendation);
  elements.haircutPreview.scrollIntoView({ behavior: "smooth", block: "start" });
}

elements.tryonBlend.addEventListener("input", () => {
  elements.tryonBlendValue.textContent = `${elements.tryonBlend.value}%`;
  if (activeHaircut) renderHaircutPreview(activeHaircut);
});

function drawCanvases(result) {
  originalImage = new Image();
  maskImage = new Image();
  originalImage.onload = () => {
    const width = originalImage.naturalWidth;
    const height = originalImage.naturalHeight;
    [elements.beforeCanvas, elements.afterCanvas].forEach((canvas) => {
      canvas.width = width;
      canvas.height = height;
    });
    sizeComparisonStage();
    beforeContext.drawImage(originalImage, 0, 0);
    selection = { x1: 0, y1: 0, x2: width, y2: height };
    maskImage.onload = () => {
      hairMaskBounds = findHairMaskBounds(width, height);
      renderColorPreview();
      renderHairMap();
    };
    maskImage.src = result.maskData;
    elements.download.disabled = false;
  };
  originalImage.src = result.imageData;
}

function findHairMaskBounds(width, height) {
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const context = maskCanvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(maskImage, 0, 0, width, height);
  const pixels = context.getImageData(0, 0, width, height).data;
  let left = width;
  let top = height;
  let right = -1;
  let bottom = -1;
  for (let pixel = 0; pixel < pixels.length; pixel += 4) {
    if (pixels[pixel] < 96) continue;
    const index = pixel / 4;
    const x = index % width;
    const y = Math.floor(index / width);
    left = Math.min(left, x);
    top = Math.min(top, y);
    right = Math.max(right, x);
    bottom = Math.max(bottom, y);
  }
  return right < left || bottom < top ? null : { left, top, right: right + 1, bottom: bottom + 1 };
}

function buildColorControls(colors) {
  elements.families.replaceChildren();
  Object.keys(paletteFamilies).forEach((family) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "family-button";
    button.textContent = family;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(family === activeFamily));
    button.addEventListener("click", () => {
      activeFamily = family;
      buildColorControls(colors);
    });
    elements.families.append(button);
  });

  elements.palette.replaceChildren();
  paletteFamilies[activeFamily].forEach((name) => {
    const rgb = colors[name];
    if (!rgb) return;
    const swatch = document.createElement("button");
    swatch.type = "button";
    swatch.className = "color-swatch";
    swatch.style.backgroundColor = `rgb(${rgb.join(",")})`;
    swatch.dataset.name = name;
    swatch.setAttribute("aria-label", name);
    swatch.setAttribute("aria-pressed", String(selectedColor.name === name));
    swatch.title = name;
    swatch.addEventListener("click", () => {
      selectedColor = { name, rgb };
      elements.selectedColorName.textContent = name;
      buildColorControls(colors);
      renderColorPreview();
    });
    elements.palette.append(swatch);
  });
}

function canvasSelection() {
  if (!selection) return { x1: 0, y1: 0, x2: elements.afterCanvas.width, y2: elements.afterCanvas.height };
  return {
    x1: Math.max(0, Math.min(selection.x1, selection.x2)),
    y1: Math.max(0, Math.min(selection.y1, selection.y2)),
    x2: Math.min(elements.afterCanvas.width, Math.max(selection.x1, selection.x2)),
    y2: Math.min(elements.afterCanvas.height, Math.max(selection.y1, selection.y2)),
  };
}

function rgbToHsl(red, green, blue) {
  const r = red / 255;
  const g = green / 255;
  const b = blue / 255;
  const maximum = Math.max(r, g, b);
  const minimum = Math.min(r, g, b);
  const difference = maximum - minimum;
  const lightness = (maximum + minimum) / 2;
  let hue = 0;
  let saturation = 0;

  if (difference > 0) {
    saturation = difference / (1 - Math.abs(2 * lightness - 1));
    if (maximum === r) hue = ((g - b) / difference + (g < b ? 6 : 0)) / 6;
    else if (maximum === g) hue = ((b - r) / difference + 2) / 6;
    else hue = ((r - g) / difference + 4) / 6;
  }

  return [hue, saturation, lightness];
}

function hslToRgb(hue, saturation, lightness) {
  const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
  const hueSection = hue * 6;
  const secondary = chroma * (1 - Math.abs((hueSection % 2) - 1));
  let red = 0;
  let green = 0;
  let blue = 0;

  if (hueSection < 1) [red, green] = [chroma, secondary];
  else if (hueSection < 2) [red, green] = [secondary, chroma];
  else if (hueSection < 3) [green, blue] = [chroma, secondary];
  else if (hueSection < 4) [green, blue] = [secondary, chroma];
  else if (hueSection < 5) [red, blue] = [secondary, chroma];
  else [red, blue] = [chroma, secondary];

  const offset = lightness - chroma / 2;
  return [
    Math.round((red + offset) * 255),
    Math.round((green + offset) * 255),
    Math.round((blue + offset) * 255),
  ];
}

function renderColorPreview() {
  if (!originalImage?.complete || !maskImage?.complete || !elements.afterCanvas.width) return;
  const width = elements.afterCanvas.width;
  const height = elements.afterCanvas.height;
  afterContext.clearRect(0, 0, width, height);
  afterContext.drawImage(originalImage, 0, 0);

  const output = afterContext.getImageData(0, 0, width, height);
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
  maskContext.drawImage(maskImage, 0, 0, width, height);
  const maskPixels = maskContext.getImageData(0, 0, width, height).data;
  const bounds = canvasSelection();
  const intensity = Number(elements.blend.value) / 100;
  const [targetHue, targetSaturation, targetLightness] = rgbToHsl(...selectedColor.rgb);
  const regionStart = activeColorRegion === "scalp" ? 0
    : activeColorRegion === "middle" ? 0.3
      : activeColorRegion === "ends" ? 0.7 : 0;
  const regionEnd = activeColorRegion === "scalp" ? 0.3
    : activeColorRegion === "middle" ? 0.7
      : activeColorRegion === "ends" ? 1 : 1;
  const regionTop = hairMaskBounds
    ? hairMaskBounds.top + (hairMaskBounds.bottom - hairMaskBounds.top) * regionStart
    : 0;
  const regionBottom = hairMaskBounds
    ? hairMaskBounds.top + (hairMaskBounds.bottom - hairMaskBounds.top) * regionEnd
    : height;
  const regionFeather = hairMaskBounds ? Math.max(1, (hairMaskBounds.bottom - hairMaskBounds.top) * 0.025) : 1;

  const regionStrengthAt = (y) => {
    const topFade = activeColorRegion === "middle" || activeColorRegion === "ends"
      ? Math.min(1, Math.max(0, (y - regionTop) / regionFeather))
      : 1;
    const bottomFade = activeColorRegion === "middle" || activeColorRegion === "scalp"
      ? Math.min(1, Math.max(0, (regionBottom - y) / regionFeather))
      : 1;
    return Math.min(topFade, bottomFade);
  };

  let luminanceTotal = 0;
  let maskStrengthTotal = 0;
  for (let y = bounds.y1; y < bounds.y2; y += 1) {
    if (y < regionTop - regionFeather || y > regionBottom + regionFeather) continue;
    const regionStrength = regionStrengthAt(y);
    if (regionStrength <= 0) continue;
    for (let x = bounds.x1; x < bounds.x2; x += 1) {
      const index = (y * width + x) * 4;
      const maskStrength = (maskPixels[index] / 255) * regionStrength;
      if (maskStrength <= 0) continue;
      const luminance = (
        output.data[index] * 0.2126
        + output.data[index + 1] * 0.7152
        + output.data[index + 2] * 0.0722
      ) / 255;
      luminanceTotal += luminance * maskStrength;
      maskStrengthTotal += maskStrength;
    }
  }

  const averageLuminance = maskStrengthTotal
    ? luminanceTotal / maskStrengthTotal
    : 0.5;
  const baseLightness = targetLightness * 0.85 + averageLuminance * 0.15;
  for (let y = bounds.y1; y < bounds.y2; y += 1) {
    if (y < regionTop - regionFeather || y > regionBottom + regionFeather) continue;
    const regionStrength = regionStrengthAt(y);
    if (regionStrength <= 0) continue;
    for (let x = bounds.x1; x < bounds.x2; x += 1) {
      const index = (y * width + x) * 4;
      const maskStrength = (maskPixels[index] / 255) * regionStrength;
      if (maskStrength <= 0) continue;
      const sourceRed = output.data[index];
      const sourceGreen = output.data[index + 1];
      const sourceBlue = output.data[index + 2];
      const sourceLuminance = (
        sourceRed * 0.2126 + sourceGreen * 0.7152 + sourceBlue * 0.0722
      ) / 255;
      const dyedLightness = Math.min(
        0.92,
        Math.max(0.06, baseLightness + (sourceLuminance - averageLuminance) * 0.65),
      );
      const dyedChannels = hslToRgb(targetHue, targetSaturation, dyedLightness);
      const colorStrength = intensity * maskStrength;
      output.data[index] = Math.round(
        sourceRed * (1 - colorStrength) + dyedChannels[0] * colorStrength,
      );
      output.data[index + 1] = Math.round(
        sourceGreen * (1 - colorStrength) + dyedChannels[1] * colorStrength,
      );
      output.data[index + 2] = Math.round(
        sourceBlue * (1 - colorStrength) + dyedChannels[2] * colorStrength,
      );
    }
  }
  afterContext.putImageData(output, 0, 0);

  const selectedPixels = Math.max(0, bounds.x2 - bounds.x1) * Math.max(0, bounds.y2 - bounds.y1);
  const totalPixels = width * height;
  const regionName = activeColorRegion === "custom"
    ? "Custom area"
    : activeColorRegion === "whole" ? "Whole hair"
      : activeColorRegion === "scalp" ? "Scalp / roots"
        : activeColorRegion === "middle" ? "Middle" : "Ends";
  elements.selectionLabel.textContent = activeColorRegion === "custom"
    ? `${regionName} · ${((selectedPixels / totalPixels) * 100).toFixed(0)}% of photo`
    : `${regionName} selected`;
  elements.colorRegionLabel.textContent = regionName.toUpperCase();
}

function setColorRegion(region) {
  activeColorRegion = region;
  if (elements.afterCanvas.width) {
    selection = { x1: 0, y1: 0, x2: elements.afterCanvas.width, y2: elements.afterCanvas.height };
  }
  elements.colorRegionButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.colorRegion === region));
  });
  renderColorPreview();
}

elements.colorRegionButtons.forEach((button) => {
  button.addEventListener("click", () => setColorRegion(button.dataset.colorRegion));
});

function pointerPosition(event) {
  const canvas = elements.afterCanvas;
  const bounds = elements.comparisonStage.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(canvas.width, Math.round((event.clientX - bounds.left) * canvas.width / bounds.width))),
    y: Math.max(0, Math.min(canvas.height, Math.round((event.clientY - bounds.top) * canvas.height / bounds.height))),
  };
}

let dragStart = null;
elements.comparisonStage.addEventListener("pointerdown", (event) => {
  if (!originalImage) return;
  activeColorRegion = "custom";
  elements.colorRegionButtons.forEach((button) => button.setAttribute("aria-pressed", "false"));
  dragStart = pointerPosition(event);
  selection = { x1: dragStart.x, y1: dragStart.y, x2: dragStart.x, y2: dragStart.y };
  elements.comparisonStage.setPointerCapture(event.pointerId);
});
elements.comparisonStage.addEventListener("pointermove", (event) => {
  if (!dragStart) return;
  const position = pointerPosition(event);
  selection = { x1: dragStart.x, y1: dragStart.y, x2: position.x, y2: position.y };
  renderColorPreview();
});
elements.comparisonStage.addEventListener("pointerup", () => {
  if (!dragStart) return;
  dragStart = null;
  renderColorPreview();
});
elements.comparisonStage.addEventListener("pointercancel", () => {
  dragStart = null;
});

elements.blend.addEventListener("input", () => {
  elements.blendValue.textContent = `${elements.blend.value}%`;
  renderColorPreview();
});

elements.download.addEventListener("click", () => {
  if (!elements.afterCanvas.width) return;
  const link = document.createElement("a");
  link.download = `style-strand-ai-${selectedColor.name.toLowerCase().replaceAll(" ", "-")}.png`;
  link.href = elements.afterCanvas.toDataURL("image/png");
  link.click();
});

function render(args) {
  window.componentArgs = args;
  if (args.result) {
    if (args.result.imageData) showUploadState(args.result.imageData, "");
    if (args.result.imageData || args.result.error) renderProfile(args.result);
  } else {
    elements.scanLine.hidden = true;
    setFrameHeight();
  }
}

window.addEventListener("message", (event) => {
  if (event.data?.type === "streamlit:render") render(event.data.args || {});
});

sendToStreamlit("streamlit:componentReady", { apiVersion: 1 });
sendToStreamlit("streamlit:setFrameHeight", { height: document.body.scrollHeight });