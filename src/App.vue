<template>
  <h4>Reference Image</h4>
  <img ref="refImg" src="./assets/rpi.png" />

  <h4>Annotation</h4>
  <img ref="annotatedImg" src="./assets/annotated.png" />

  <h4>Annotation combined Image</h4>
  <div class="overlay border">
    <img src="./assets/rpi.png" />
    <img src="./assets/annotated.png" />
  </div>

  <h4>Target Image</h4>
  <img class="border" ref="targetImg" src="./assets/rpi2.png" />

  <h4>Feature matches</h4>
  <canvas ref="canvas1" class="canvas" />

  <h4>Final warping</h4>
  <p>
    <span>TickTime: {{ tickTime }}ms</span> <span>Frame#: {{ frameNum }}</span>
  </p>
  <div class="overlay">
    <img src="./assets/rpi2.png" />
    <canvas ref="canvas2" />
  </div>
</template>

<script>
/**
 * https://scottsuhy.com/2021/02/01/image-alignment-feature-based-in-opencv-js-javascript/
 */

import cv from "opencv-ts";

let orb;
let keypoints1;
let descriptors1;

export default {
  name: "App",
  components: {},
  data() {
    return {
      tickTime: -1,
      frameNum: 0,
      tickTimer: -1,
    };
  },
  methods: {
    tick() {
      this.frameNum++;
      const tickStart = Date.now();
      //im2 is the video feed
      const im2 = cv.imread(this.$refs.targetImg);
      //im3 is annotated png
      const im3 = cv.imread(this.$refs.annotatedImg);

      const im2Gray = new cv.Mat();
      cv.cvtColor(im2, im2Gray, cv.COLOR_BGR2GRAY);

      const keypoints2 = new cv.KeyPointVector();
      const descriptors2 = new cv.Mat();

      const mask = new cv.Mat();
      // 46.3% of total compute time
      orb.detectAndCompute(im2Gray, mask, keypoints2, descriptors2);

      // Match features.
      const bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
      const matches = new cv.DMatchVector();
      // 31.5% of total compute time
      bf.match(descriptors1, descriptors2, matches);

      // Sort matches by score
      const goodMatches = new cv.DMatchVector();
      for (let i = 0; i < matches.size(); i++) {
        if (matches.get(i).distance < 50) {
          goodMatches.push_back(matches.get(i));
        }
      }

      // Draw top matches (optional)
      // const imMatches = new cv.Mat();
      // const color = new cv.Scalar(0, 255, 0, 255);
      // cv.drawMatches(
      //   im1,
      //   keypoints1,
      //   im2,
      //   keypoints2,
      //   goodMatches,
      //   imMatches,
      //   color
      // );
      // cv.imshow(this.$refs.canvas1, imMatches);

      // Extract location of good matches
      const points1 = [];
      const points2 = [];

      for (let i = 0; i < goodMatches.size(); i++) {
        points1.push(keypoints1.get(goodMatches.get(i).queryIdx).pt.x);
        points1.push(keypoints1.get(goodMatches.get(i).queryIdx).pt.y);
        points2.push(keypoints2.get(goodMatches.get(i).trainIdx).pt.x);
        points2.push(keypoints2.get(goodMatches.get(i).trainIdx).pt.y);
      }

      // Find homography
      const mat1 = cv.matFromArray(points1.length, 2, cv.CV_32F, points1);
      const mat2 = cv.matFromArray(points2.length, 2, cv.CV_32F, points2);
      const h = cv.findHomography(mat1, mat2, cv.RANSAC);

      // Use homography to warp image
      const finalResult = new cv.Mat();
      cv.warpPerspective(im3, finalResult, h, im2.size());
      cv.imshow(this.$refs.canvas2, finalResult);

      this.tickTime = Date.now() - tickStart;

      mat1.delete();
      mat2.delete();
      mask.delete();
      h.delete();
      keypoints2.delete();
      descriptors2.delete();
      goodMatches.delete();
      bf.delete();
      finalResult.delete();
      im2Gray.delete();
      im2.delete();
      im3.delete();

      requestAnimationFrame(this.tick);
    },
  },
  mounted() {
    // https://stackoverflow.com/questions/65855110/how-can-i-align-images-using-opencv-js
    cv.onRuntimeInitialized = () => {
      orb = new cv.ORB(1000);

      //im1 is the reference image we are trying to align
      const im1 = cv.imread(this.$refs.refImg);

      // Convert images to grayscale
      const im1Gray = new cv.Mat();
      cv.cvtColor(im1, im1Gray, cv.COLOR_BGR2GRAY);

      // Variables to store keypoints and descriptors
      keypoints1 = new cv.KeyPointVector();
      descriptors1 = new cv.Mat();

      // Detect ORB features and compute descriptors.
      const mask = new cv.Mat();
      orb.detectAndCompute(im1Gray, mask, keypoints1, descriptors1);

      this.ticktimer = requestAnimationFrame(this.tick);

      // clean up
      im1.delete();
      im1Gray.delete();
      mask.delete();
      // matches.delete();
      // bf.delete();
      // orb.delete();
      // descriptors1.delete();
      // descriptors2.delete();
      // keypoints1.delete();
      // keypoints2.delete();
      // im1Gray.delete();
      // im2Gray.delete();
      // h.delete();
      // finalResult.delete();
      // mat1.delete();
      // mat2.delete();
    };
  },
  beforeUnmount() {
    cancelAnimationFrame(this.tickTimer);
    orb.delete();
    keypoints1.delete();
    descriptors1.delete();
  },
};
</script>

<style>
* {
  box-sizing: border-box;
}

.canvas {
  border: 3px solid black;
}

.border {
  border: 1px solid black;
}

.overlay {
  position: relative;
}

.overlay *:first-child {
  position: relative;
  top: 0;
  left: 0;
}

.overlay *:not(:first-child) {
  position: absolute;
  top: 0;
  left: 0;
}
</style>
