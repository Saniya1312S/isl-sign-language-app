/**
 * Camera Service - Handles MediaDevices webcam access, frame capture, and cleanup.
 */
class CameraService {
  constructor(videoElementId) {
    this.videoEl = document.getElementById(videoElementId);
    this.stream = null;
    this.isActive = false;
  }

  async startCamera(constraints = { video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" } }) {
    if (this.isActive && this.stream) {
      return true;
    }

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera API (getUserMedia) not supported in this browser.");
      }

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (this.videoEl) {
        this.videoEl.srcObject = this.stream;
        await this.videoEl.play();
      }
      this.isActive = true;
      return true;
    } catch (err) {
      console.error("Camera access failed:", err);
      this.isActive = false;
      this.stream = null;
      throw err;
    }
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.videoEl) {
      this.videoEl.srcObject = null;
    }
    this.isActive = false;
  }

  captureFrameAsBase64(format = "image/jpeg", quality = 0.85) {
    if (!this.isActive || !this.videoEl) {
      return null;
    }

    const video = this.videoEl;
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");

    // Capture standard frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL(format, quality);
  }
}

// Global camera service instances for Page 1 and Page 2
const page1Camera = new CameraService("p1-video");
const page2Camera = new CameraService("p2-video");
