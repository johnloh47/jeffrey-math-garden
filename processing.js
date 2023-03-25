var model;

async function loadModel() {
    
    model = await tf.loadGraphModel('TFJS/model.json');
    
}

function predictImage() {
    // console.log('processing.....');

    // Load image and convert to black and white
    let image = cv.imread(canvas);
    cv.cvtColor(image, image, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(image, image, 175, 255, cv.THRESH_BINARY);

    // find contours of the image
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat()
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);

    // Calculate Bounding Rectangle and crop image
    let cnt = contours.get(0);
    let rect = cv.boundingRect(cnt);
    image = image.roi(rect);

    // calculate new size 
    var height = image.rows;
    var width = image.cols;

    if (height > width) {
        height = 20;
        const scaleFactor = image.rows / height;
        width = Math.round(image.cols / scaleFactor);
    } else {
        width = 20;
        const scaleFactor = image.cols / width;
        height = Math.round(image.rows / scaleFactor);
    }

    // Resize
    let newSize = new cv.Size(width, height);
    cv.resize(image, image, newSize, 0, 0, cv.INTER_AREA);

    // Add padding to become 28 x 28 pixels
    const LEFT = Math.ceil(4 + (20 - width)/2); // round up
    const RIGHT = Math.floor(4 + (20 - width)/2); // round down
    const TOP = Math.ceil(4 + (20 - height)/2);
    const BOTTOM = Math.floor(4 + (20 - height)/2);

    // console.log(`top: ${TOP}, bottom: ${BOTTOM}, left: ${LEFT}, right: ${RIGHT}`);

    const BLACK  = new cv.Scalar(0, 0, 0, 0); // color black
    cv.copyMakeBorder(image, image, TOP, BOTTOM, LEFT, RIGHT, cv.BORDER_CONSTANT, BLACK);

    // Center of Mass
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    cnt = contours.get(0);
    const Moments = cv.moments(cnt, false);

    const cx = Moments.m10 / Moments.m00;
    const cy = Moments.m01 / Moments.m00;
    // console.log(`M00: ${Moments.m00}, cx: ${cx}, cy: ${cy}`);

    // Shift the image to center
    const X_SHIFT = Math.round(image.cols/2 - cx);
    const Y_SHIFT = Math.round(image.rows/2 - cy);

    newSize  = new cv.Size(image.cols, image.rows);
    const M = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, X_SHIFT, 0, 1, Y_SHIFT]);
    cv.warpAffine(image, image, M, newSize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, BLACK);

    // Normalize the image pixels by dividing 255.0
    let pixelValues = image.data;
    // console.log(`pixel values: ${pixelValues}`);

    pixelValues = Float32Array.from(pixelValues);

    pixelValues = pixelValues.map(function(item) {
        return item / 255.0;
    });

    // console.log(`scaled array: ${pixelValues}`);

    const X = tf.tensor([pixelValues]);
    // console.log(`Shape of Tensor: ${X.shape}`);
    // console.log(`dtype of Tensor: ${X.dtype}`);

    const result = model.predict(X);
    result?.print();

    // console.log(tf.memory());

    const output = result.dataSync()[0];


    // testing Only (delete later)
    // const outputCanvas = document.createElement('CANVAS');
    // cv.imshow(outputCanvas, image);
    // document.body.appendChild(outputCanvas);

    //Cleanup
    image.delete();
    contours.delete();
    hierarchy.delete();
    cnt.delete();
    M.delete();
    X.dispose();
    result.dispose();

    return output;
    

}
