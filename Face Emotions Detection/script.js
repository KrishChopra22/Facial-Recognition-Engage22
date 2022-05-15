// defining the video element, from html
const video = document.getElementById('video')

// Promise ensures that video will be only started after loading all the models
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models')
]).then(startVideo).catch(err => console.error(err))

// function to start the video, by getting the media(video, streaming)
function startVideo() {
  navigator.mediaDevices.getUserMedia(
    { video: true }).then(
      stream => video.srcObject = stream,
      err => console.error(err)
    )
}

/* when the event-listener was'play', it wanted to immediately started playing the video, 
when loading of models was just completed and it throws error, hence 'playing' is used as other alternative */
video.addEventListener('playing', () => {
  const canvas = faceapi.createCanvasFromMedia(video)
  document.body.append(canvas)
  const displaySize = { width: video.width, height: video.height }

  // to match the dimentions of the pre-defined canvas and the size given to detection
  faceapi.matchDimensions(canvas, displaySize)
  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions()

    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    // faceapi.draw.drawDetections(canvas, resizedDetections)   //to draw detections(rectangle)
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)   //to draw the landmark points(accuracy in detection)
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections)   //to detect expressions(happy/sad/neutral) with (0 <= [probability of accuracy] <= 1)
  }, 100)
})