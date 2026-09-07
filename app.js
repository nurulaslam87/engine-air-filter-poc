import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_PATH = "./best.onnx";
const IMAGE_SIZE = 224;

// Folder/class order used during training: BAD=0, GOOD=1
const CLASS_NAMES = ["BAD", "GOOD"];

const AIR_FILTER_LABEL = "a photo of a car engine air filter";
const GATE_LABELS = [
  AIR_FILTER_LABEL,
  "a photo of another automotive component",
  "a photo of a non-automotive object"
];

const input = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const button = document.getElementById("inspectBtn");
const status = document.getElementById("status");

let selectedFile = null;
let selectedImage = null;
let yoloSession = null;
let clipGate = null;

input.addEventListener("change", async (event) => {
  selectedFile = event.target.files?.[0] ?? null;

  if (!selectedFile) {
    selectedImage = null;
    preview.style.display = "none";
    status.textContent = "Waiting for an image.";
    return;
  }

  const objectUrl = URL.createObjectURL(selectedFile);
  preview.src = objectUrl;
  preview.style.display = "block";
  selectedImage = await loadImage(objectUrl);
  status.textContent = "Image ready. Tap Inspect Filter.";
});

button.addEventListener("click", async () => {
  if (!selectedFile || !selectedImage) {
    status.textContent = "Please take or upload a photo first.";
    return;
  }

  button.disabled = true;

  try {
    await loadModels();

    status.textContent = "Step 1 of 2: Checking whether this is an engine air filter...";
    const gateResults = await clipGate(selectedFile, GATE_LABELS);
    const topGate = gateResults[0];

    if (topGate.label !== AIR_FILTER_LABEL) {
      status.textContent =
        "NOT AN ENGINE AIR FILTER\n\n" +
        `Gate result: ${topGate.label}\n` +
        `Gate confidence: ${(topGate.score * 100).toFixed(1)}%`;
      return;
    }

    status.textContent = "Step 2 of 2: Evaluating filter condition...";
    const tensor = imageToTensor(selectedImage);
    const inputName = yoloSession.inputNames[0];
    const outputMap = await yoloSession.run({ [inputName]: tensor });
    const outputName = yoloSession.outputNames[0];
    const rawScores = Array.from(outputMap[outputName].data);
    const probs = normalizeProbabilities(rawScores);

    const classId = probs[1] > probs[0] ? 1 : 0;
    const prediction = CLASS_NAMES[classId];
    const confidence = probs[classId] * 100;

    status.textContent =
      "ENGINE AIR FILTER DETECTED\n\n" +
      `Condition: ${prediction}\n` +
      `Confidence: ${confidence.toFixed(1)}%\n` +
      `Air-filter gate: ${(topGate.score * 100).toFixed(1)}%`;
  } catch (error) {
    console.error(error);
    status.textContent =
      "Something went wrong.\n\n" +
      error.message +
      "\n\nOpen the browser console for more details.";
  } finally {
    button.disabled = false;
  }
});

async function loadModels() {
  if (!yoloSession) {
    status.textContent = "Loading your air-filter model...";
    yoloSession = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"]
    });
  }

  if (!clipGate) {
    status.textContent =
      "Loading the image-verification model for the first time.\n" +
      "This can take a while on a phone because it must download the pretrained model.";

    clipGate = await pipeline(
      "zero-shot-image-classification",
      "Xenova/clip-vit-base-patch32"
    );
  }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function imageToTensor(img) {
  const srcW = img.naturalWidth || img.width;
  const srcH = img.naturalHeight || img.height;

  // Match Ultralytics classification inference:
  // resize shortest edge to 224, preserve aspect ratio, then center-crop 224x224.
  const scale = IMAGE_SIZE / Math.min(srcW, srcH);
  const resizedW = Math.round(srcW * scale);
  const resizedH = Math.round(srcH * scale);

  const temp = document.createElement("canvas");
  temp.width = resizedW;
  temp.height = resizedH;
  const tempCtx = temp.getContext("2d", { willReadFrequently: true });
  tempCtx.imageSmoothingEnabled = true;
  tempCtx.imageSmoothingQuality = "high";
  tempCtx.drawImage(img, 0, 0, resizedW, resizedH);

  const sx = Math.floor((resizedW - IMAGE_SIZE) / 2);
  const sy = Math.floor((resizedH - IMAGE_SIZE) / 2);

  const crop = document.createElement("canvas");
  crop.width = IMAGE_SIZE;
  crop.height = IMAGE_SIZE;
  const cropCtx = crop.getContext("2d", { willReadFrequently: true });
  cropCtx.drawImage(temp, sx, sy, IMAGE_SIZE, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const pixels = cropCtx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  const floatData = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);
  const plane = IMAGE_SIZE * IMAGE_SIZE;

  // YOLO26 classification uses RGB values scaled to 0..1.
  // Do NOT apply ImageNet mean/std normalization here.
  for (let i = 0; i < plane; i++) {
    const px = i * 4;
    floatData[i] = pixels[px] / 255;
    floatData[plane + i] = pixels[px + 1] / 255;
    floatData[2 * plane + i] = pixels[px + 2] / 255;
  }

  return new ort.Tensor("float32", floatData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

function normalizeProbabilities(values) {
  const allBetweenZeroAndOne = values.every(v => v >= 0 && v <= 1);
  const sum = values.reduce((a, b) => a + b, 0);

  if (allBetweenZeroAndOne && Math.abs(sum - 1) < 0.05) {
    return values;
  }

  const maxValue = Math.max(...values);
  const exps = values.map(v => Math.exp(v - maxValue));
  const expSum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / expSum);
}
