/*
File: index.js
Project: Motion Predictor
@author: Conner Santa Monica

Description:
Handles frontend logic and interactions for the Motion Predictor UI.
Includes event listeners for upload, webcam selection, 
and real-time video source updates from the FastAPI backend.

Endpoints:
- /upload_video
- /set_source
*/

async function setSource(source) {
  const formData = new FormData();
  formData.append("source", source);

  const response = await fetch("/set_source", {
    method: "POST",
    body: formData,
  });

  let result;
  try {
    result = await response.json();
  } catch {
    updateStatus("Invalid response from server");
    return;
  }

  const msg = result.message || result.error || "Unknown response";
  updateStatus(msg);
}

document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(event.target);

  updateStatus("⏳ Uploading video...");
  const response = await fetch("/upload_video", {
    method: "POST",
    body: formData,
  });

  let result;
  try {
    result = await response.json();
  } catch {
    updateStatus("Upload failed: invalid server response");
    return;
  }

  const msg = result.message || result.error || "Unknown response";
  updateStatus(msg);
});

function updateStatus(text) {
  const status = document.getElementById("status");
  status.textContent = text;
}
