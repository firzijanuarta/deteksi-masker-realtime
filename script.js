let modelMask, modelFace;
const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");
const startBtn = document.getElementById("startBtn");

if (startBtn) {
  startBtn.addEventListener("click", () => {
    // 1. Sembunyikan layar welcome
    document.getElementById("welcomeScreen").style.display = "none";
    // 2. Tampilkan area kamera
    document.getElementById("detectorScreen").style.display = "flex";
    // 3. Eksekusi pemanggilan AI
    init();
  });
}
const TILT_THRESHOLD_DERAJAT = 15;

// 1. Setup Webcam (tidak berubah)
async function setupWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        video.play();
        resolve(video);
      };
    });
  } catch (error) {
    statusText.innerText = "Gagal mengakses Webcam! Izinkan akses kamera.";
    console.error(error);
  }
}

// 2. Cek Posisi Masker — DIPERBAIKI: ROI menyesuaikan ukuran wajah
function cekPosisiMasker(videoElement, landmarks) {
  const rightEye = landmarks[0];
  const leftEye = landmarks[1];
  const nose = landmarks[2];
  const mouth = landmarks[3];

  // Lebar ROI: dari mata kiri ke mata kanan (otomatis menyesuaikan jarak wajah ke kamera)
  const x1 = Math.min(leftEye[0], rightEye[0]);
  const x2 = Math.max(leftEye[0], rightEye[0]);

  // Tinggi ROI: dari sedikit di atas hidung sampai mulut (area yang seharusnya tertutup masker)
  const y1 = nose[1] - (mouth[1] - nose[1]) * 0.3;
  const y2 = mouth[1];

  const roiW = Math.max(10, x2 - x1);
  const roiH = Math.max(10, y2 - y1);

  const sampleCanvas = document.createElement("canvas");
  const sampleCtx = sampleCanvas.getContext("2d");
  sampleCanvas.width = roiW;
  sampleCanvas.height = roiH;

  sampleCtx.drawImage(videoElement, x1, y1, roiW, roiH, 0, 0, roiW, roiH);
  const imgData = sampleCtx.getImageData(0, 0, roiW, roiH).data;

  let skinPixels = 0;
  const totalPixels = roiW * roiH;

  for (let i = 0; i < imgData.length; i += 4) {
    const r = imgData[i];
    const g = imgData[i + 1];
    const b = imgData[i + 2];

    if (r > 95 && g > 40 && b > 20 && r > g && r > b && Math.abs(r - g) > 15) {
      skinPixels++;
    }
  }

  const skinRatio = skinPixels / totalPixels;
  return skinRatio < 0.4;
}

// 3. BARU: Cek Kemiringan (sebelumnya tidak ada di kode lama)
function cekKemiringan(landmarks) {
  const rightEye = landmarks[0];
  const leftEye = landmarks[1];

  const dx = leftEye[0] - rightEye[0];
  const dy = leftEye[1] - rightEye[1];
  const angleDeg = Math.atan2(dy, dx) * (180 / Math.PI);

  return { ok: Math.abs(angleDeg) < TILT_THRESHOLD_DERAJAT, angle: angleDeg };
}

// 4. Inisialisasi (tidak berubah)
async function init() {
  try {
    statusText.innerText = "Memuat Model Deteksi Wajah...";
    modelFace = await blazeface.load();

    statusText.innerText = "Memuat Model Deteksi Masker...";
    modelMask = await tf.loadLayersModel("tfjs_model/model.json");

    statusText.innerText = "Menyalakan Kamera...";
    await setupWebcam();

    statusText.innerText = "Sistem Siap! Melakukan Deteksi...";
    statusText.style.color = "green";

    detectFrame();
  } catch (error) {
    statusText.innerText = "Gagal memuat sistem AI!";
    console.error(error);
  }
}

// 5. Loop Deteksi Utama — DIPERBAIKI: gabungkan posisi + kemiringan
async function detectFrame() {
  const predictions = await modelFace.estimateFaces(video, false);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (predictions.length > 0) {
    for (let i = 0; i < predictions.length; i++) {
      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];

      const x = start[0];
      const y = start[1];
      const w = size[0];
      const h = size[1];

      const landmarks = predictions[i].landmarks; // [rightEye, leftEye, nose, mouth, rightEar, leftEar]

      if (x < 0 || y < 0 || x + w > video.width || y + h > video.height)
        continue;

      const resultTensor = tf.tidy(() => {
        const imgTensor = tf.browser.fromPixels(video);
        const cropped = imgTensor.slice(
          [parseInt(y), parseInt(x), 0],
          [parseInt(h), parseInt(w), 3],
        );
        const resized = tf.image.resizeBilinear(cropped, [224, 224]); // diganti dari resizeNearestNeighbor -> lebih konsisten dengan cv2.resize di Python
        const normalized = resized
          .toFloat()
          .sub(tf.scalar(127.5))
          .div(tf.scalar(127.5));
        return normalized.expandDims(0);
      });

      const prediction = await modelMask.predict(resultTensor).data();
      const confidence = prediction[0];
      resultTensor.dispose();

      let label = "";
      let color = "";
      let scorePercent = 0;

      if (confidence > 0.5) {
        label = "Tanpa Masker";
        color = "#FF0000";
        scorePercent = confidence * 100;
      } else {
        scorePercent = (1 - confidence) * 100;

        const posisiOk = cekPosisiMasker(video, landmarks);
        const tilt = cekKemiringan(landmarks);

        if (posisiOk && tilt.ok) {
          label = "Masker - Rapi";
          color = "#00FF00";
        } else {
          label = "Masker - Belum Rapi";
          color = "#FFFF00";
        }
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, w, h);

      ctx.fillStyle = color;
      ctx.font = "bold 16px Arial";
      ctx.textAlign = "center";

      const textToDraw = `${label} (${scorePercent.toFixed(0)}%)`;
      const textX = x + w / 2;
      const textY = y > 20 ? y - 10 : y + 20;

      ctx.save();
      ctx.translate(textX, textY);
      ctx.scale(-1, 1);
      ctx.fillText(textToDraw, 0, 0);
      ctx.restore();
    }
  }

  requestAnimationFrame(detectFrame);
}
