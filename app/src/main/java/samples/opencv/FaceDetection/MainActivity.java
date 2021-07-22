package samples.opencv.FaceDetection;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import org.opencv.core.Core;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {

    private static final String TAG = "OpenCV/Sample/MobileNet";
    private CameraBridgeViewBase mOpenCvCameraView;
    private CascadeClassifier faceCas;

    // Initialize OpenCV manager.
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };
    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // Set up camera listener.
        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCameraIndex(1);
        mOpenCvCameraView.enableFpsMeter();
        mOpenCvCameraView.setCvCameraViewListener(this);

    }
    // Load a network.
    public void onCameraViewStarted(int width, int height) {
        String face = getPath("face_data.xml", this);
        faceCas = new CascadeClassifier(face);

        //net = Dnn.readNetFromCaffe(proto,weights);

        Log.i(TAG, "Network loaded successfully");
    }
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        final double IN_SCALE_FACTOR = 0.007843;
        final double MEAN_VAL = 127.5;
        final double THRESHOLD = 0.2;
        // Get a new frame

        Mat frame = inputFrame.rgba();

        Core.transpose(frame, frame);
        Core.flip(frame, frame, -1);

        MatOfRect faces = new MatOfRect();
        faceCas.detectMultiScale(frame, faces, 1.1, 3, 0, new Size(120,120));

        for (Rect rect : faces.toArray()) {
            Imgproc.putText(frame, "Face", new Point(rect.x,rect.y-5), 1, 2, new Scalar(0,0,255));
            Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 100, 0),3);
        }

        return frame;
    }
    public void onCameraViewStopped() {}
    // Upload file to storage and return a path.
    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }

}