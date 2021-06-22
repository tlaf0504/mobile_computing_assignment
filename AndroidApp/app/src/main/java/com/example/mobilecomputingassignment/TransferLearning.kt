package com.example.mobilecomputingassignment

import android.content.Context
import android.content.DialogInterface
import android.content.Intent
import android.content.res.AssetManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.Message
import android.view.View
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.example.mobilecomputingassignment.ml.ConvertedBaseModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.w3c.dom.Text
import java.io.InputStream
import java.text.DateFormat
import java.util.*
import java.util.concurrent.locks.ReentrantLock
import kotlin.math.absoluteValue

// Type combining timestamps and sensor-channels.
// Just syntactic sugar.
typealias SensorData = Pair<Array<Long>, Array<Array<Double>>>

class TransferLearning : AppCompatActivity(),
    SensorEventListener,
    CompoundButton.OnCheckedChangeListener,
    View.OnClickListener,
    AdapterView.OnItemSelectedListener{

    private lateinit var sensorManager: SensorManager;
    private lateinit var gyroSensor: Sensor;
    private lateinit var accelSensor: Sensor;
    private lateinit var classificationThread: TFLClassificationThread;
    private lateinit var classificationTimer: Timer;
    private lateinit var classificationTimerTask: TFLClassificationTimerTask;
    private lateinit var tflDataCapturingThread: TFLDataCapturingThread;
    private lateinit var bReturnToMainButton: Button;
    private lateinit var bDebug: Button;
    private lateinit var spActivitySelectionSpinner: Spinner;
    private lateinit var lossTextView: TextView;

    var tflActivitySpinnerSelectedItemString: String = "Walking";
    var tflActivitySpinnerSelectedItemPos: Int = 0;

    private lateinit var captureDataSwitch: Switch;
    private var captureDataSwitchState:Boolean = false;
    private var captureDataSwitchStateLock = ReentrantLock()

    private lateinit var inferenceSwitch: Switch;
    private var inferenceSwitchState:Boolean = false;

    private lateinit var transferLearningSwitch: Switch;
    private var transferLearningSwitchState: Boolean = false;

    private lateinit var transferLearningCaptureModeSwitch: Switch;
    var transferLearningCaptureModeSwitchState: Boolean = false;

    // If 0: Neither data-capturing nor inference are active
    // If 1: Inference is active. No data capturing possible.
    // If 2: Data capturing active. No inference possible.
    private val appState: Int = 0

    /* ========== Sensor-data arrays ========== */

    // Assuming a sensor event occurring approximately every 150 ms, the buffer below is able to
    // hold approximately 150s of data for each activity. This should be far enough for both,
    // classification and transfer learning.
    val arraySize:Int = 10_000
    val gyroTimeArray = Array<Long>(size=arraySize, init={-1})
    val gyroSensorArray = arrayOf(
        Array<Double>(size=arraySize, init={0.0}),
        Array<Double>(size=arraySize, init={0.0}),
        Array<Double>(size=arraySize, init={0.0})
    );
    val accelTimeArray = Array<Long>(size = arraySize, init={-1})
    val accelSensorArray = arrayOf(
        Array<Double>(size=arraySize, init={0.0}),
        Array<Double>(size=arraySize, init={0.0}),
        Array<Double>(size=arraySize, init={0.0})
    );

    // Arrays for storing the transfer-learning samples
    var walkingTransferLearningSamples = mutableListOf<Array<Array<Double>>>()
    var walkingDownstairsTransferLearningSamples = mutableListOf<Array<Array<Double>>>()
    var walkingUpstairsTransferLearningSamples = mutableListOf<Array<Array<Double>>>()
    var runningTransferLearningSamples = mutableListOf<Array<Array<Double>>>()

    // After capturing samples for one activity, the corresponding timestamp in milliseconds will be
    // stored here. See https://developer.android.com/reference/java/util/Date#getTime() for further
    // information.
    val transferLearningSampleCounts: Array<Int> = arrayOf(0,0,0,0)

    lateinit var transferLearningModel: TransferLearningModelWrapper;


    //Handler triggered after model-inference
    lateinit var probabilityUpdateHandler: Handler;
    // Handler triggered after capturing transfer-learning samples for one activity
    lateinit var transferLearningActivitySamplingFinishedHandler: Handler;
    // Handler triggered after transfer-learning has completed
    lateinit var transferLearningFinishedUpdateHandler: Handler;
    lateinit var printLearningEpochHandler: Handler;
    lateinit var debugHandler: Handler;

    lateinit var activity_labels: Array<String>;

    // A lock acquired while writing data to the sensor-data arrays.
    // Acquire to lock data capturing.
    val writeDataLock = ReentrantLock()

    // Acquire to lock classification.
    val classificationLock = ReentrantLock()

    // The condition below is triggered when the user toggles the data-capturing switch in the UI.
    val transferLearningDataCapturingLock = ReentrantLock()
    val transferLearningDataCapturingCondition = transferLearningDataCapturingLock.newCondition()


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_transfer_learning)

        activity_labels = resources.getStringArray(R.array.transfer_learning_spinner_entries)
        /* Initialization-code taken from
         * https://developer.android.com/guide/topics/ui/controls/spinner (2021-04-07)
         */
        spActivitySelectionSpinner = findViewById(R.id.tfl_activity_sample_spinner)
        spActivitySelectionSpinner.onItemSelectedListener = this // Register Listener

        ArrayAdapter.createFromResource(
            this,
            R.array.transfer_learning_spinner_entries,
            android.R.layout.simple_spinner_item
        ).also {adapter ->
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            spActivitySelectionSpinner.adapter = adapter}



        probabilityUpdateHandler = Handler(mainLooper, Handler.Callback { msg -> probabilityUpdateHandlerCallback(msg) })
        transferLearningActivitySamplingFinishedHandler =
            Handler(mainLooper, Handler.Callback { msg -> transferLearningActivitySamplingFinishedHandlerCallback(msg) })
        transferLearningFinishedUpdateHandler =
            Handler(mainLooper, Handler.Callback { msg -> transferLearningFinishedUpdateHandlerCallback(msg) })
        printLearningEpochHandler = Handler(mainLooper, Handler.Callback { msg -> printLearningLossHandlerCallback(msg) })
        debugHandler = Handler(mainLooper, Handler.Callback { msg -> debugButtonCallback(msg) })

        bReturnToMainButton = findViewById(R.id.button_transfer_learning_return);
        bReturnToMainButton.setOnClickListener(this)
        captureDataSwitch = findViewById(R.id.tfl_recording_switch)
        captureDataSwitch.setOnCheckedChangeListener(this)
        inferenceSwitch = findViewById(R.id.tfl_inference_switch)
        inferenceSwitch.setOnCheckedChangeListener(this)
        transferLearningSwitch = findViewById(R.id.tfl_learning_switch)
        transferLearningSwitch.setOnCheckedChangeListener(this)
        transferLearningCaptureModeSwitch = findViewById(R.id.tfl_recording_mode_switch)
        transferLearningCaptureModeSwitch.setOnCheckedChangeListener(this)
        bDebug = findViewById(R.id.tfl_debug)
        bDebug.setOnClickListener(this)
        lossTextView = findViewById(R.id.loss_value)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager;
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        // Disable classification by default
        classificationLock.lock()

        classificationThread = TFLClassificationThread(this);
        classificationTimer = Timer();
        classificationTimerTask = TFLClassificationTimerTask(classificationThread, this);

        classificationThread.start()

        tflDataCapturingThread = TFLDataCapturingThread(this)
        tflDataCapturingThread.start()

        initUI()

        transferLearningModel = TransferLearningModelWrapper(this)

    }

    override fun onStart() {
        super.onStart()

        // Register the event-listeners for the Gyroscope and the Accelerometer.
        sensorManager.registerListener(this, gyroSensor, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, accelSensor, SensorManager.SENSOR_DELAY_NORMAL);

        // Schedule the classification task at a fixed rate of 2 seconds, but wait for 1 second
        // to start to ensure the UI is build up.
        classificationTimer.scheduleAtFixedRate(classificationTimerTask, 1000, 1000);
    }

    override fun onStop() {
        super.onStop()
        sensorManager.unregisterListener(this, gyroSensor);
        sensorManager.unregisterListener(this, accelSensor);

        // Stop triggering the classification. Be aware the the computation-task is still
        // running in the background, but after its current run it waits until a computation-cycle
        // is triggered by the timer-task. (Which will never occur as long as the timer-task is
        // not running.)
        classificationTimerTask.cancel()
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) {
            return;
        }

        val sensorType: Int = event.sensor.type;

        if (sensorType == Sensor.TYPE_GYROSCOPE) {
            val gyroX : Double = event.values[0].toDouble();
            val gyroY : Double = event.values[1].toDouble();
            val gyroZ : Double = event.values[2].toDouble();

            writeDataLock.lock()
            try {
                pushSensorValuesToBuffers(
                    gyroSensorArray,
                    gyroTimeArray,
                    event.timestamp,
                    arrayOf(gyroX, gyroY, gyroZ))

            } finally {
                writeDataLock.unlock()
            }

        } else if (sensorType == Sensor.TYPE_ACCELEROMETER) {
            val accelX: Double = event.values[0].toDouble();
            val accelY: Double = event.values[1].toDouble();
            val accelZ: Double = event.values[2].toDouble();
            // Get the timestamp in seconds
            val time: Double = event.timestamp.toDouble() * 1e-9F;

            writeDataLock.lock()
            try {
                pushSensorValuesToBuffers(
                    accelSensorArray,
                    accelTimeArray,
                    event.timestamp,
                    arrayOf(accelX, accelY, accelZ))
            } finally {
                writeDataLock.unlock()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Mandatory implementation by child-class of SensorEventListener()
    }

    private fun <T: Number> shiftRightVector(vector: Array<T>, N: Int)  {
        for (k in 0 until N) {
            vector[N - k] = vector[N - k - 1]
        }
    }

    private fun shiftRightBufferEntries(timeBuffer: Array<Long>,
                                        sensorBuffer: Array<Array<Double>>) {
        val N_columns: Int = sensorBuffer.size
        val N_entries: Int = timeBuffer.size


        // Get the number of non-empty rows
        var N_filled_entries = 0
        for (k in 0 until N_entries) {
            if (timeBuffer[k] < 0) {
                N_filled_entries = k
                break;
            }
        }

        // Shift right time- and sensor-data-arrays
        shiftRightVector(timeBuffer, N_filled_entries)
        for (k in 0 until N_columns) {
            shiftRightVector(sensorBuffer[k], N_filled_entries)
        }
    }

    private fun pushSensorValuesToBuffers(
        sensorBuffer: Array<Array<Double>>,
        timeBuffer: Array<Long>,
        timestamp: Long,
        values: Array<Double>) {

        shiftRightBufferEntries(timeBuffer, sensorBuffer)
        timeBuffer[0] = timestamp
        for (k in 0 until values.size) {
            sensorBuffer[k][0] = values[k];
        }
    }

    private fun initUI() {

        // Set inference elements
        val activities: Array<Int> = arrayOf(0,1,2,3)
        val values: Array<Array<Double>> = arrayOf(
            Array<Double>(activities.size, init = {0.0}),
            Array<Double>(activities.size, init = {0.0}))

        setActivityProbabilities(activities, values)

        // Set transfer-learning elements
        setTransferLearningActivitySamplingStats(0)
        setTransferLearningActivitySamplingStats(1)
        setTransferLearningActivitySamplingStats(2)
        setTransferLearningActivitySamplingStats(3)

        // Set the time-strings to "Never"
        for (k in 0..3) {
            val tw_id_center: Int = resources.getIdentifier("tfl_timestamps_%d_center".format(k + 1), "id", packageName)
            val tw_center: TextView = findViewById(tw_id_center)

            tw_center.text = "Never"
        }
        
        updateLossValue(Double.NaN)
    }

    // This method is NOT thread-safe!
    fun flushSensorDataBuffers() {

        gyroTimeArray.fill(-1)
        gyroSensorArray[0].fill(0.0)
        gyroSensorArray[1].fill(0.0)
        gyroSensorArray[2].fill(0.0)

        accelTimeArray.fill(-1)
        accelSensorArray[0].fill(0.0)
        accelSensorArray[1].fill(0.0)
        accelSensorArray[2].fill(0.0)
    }

    fun probabilityUpdateHandlerCallback(msg: Message):Boolean {
        val msg_converted = msg.obj as Pair<Array<Int>, Array<Array<Double>>>
        setActivityProbabilities(msg_converted.first, msg_converted.second)
        return true
    }

    fun transferLearningActivitySamplingFinishedHandlerCallback(msg: Message): Boolean {
        val (id, N_Samples) = msg.obj as Pair<Int, Int>
        setTransferLearningActivitySamplingStats(id)
        return true
    }

    fun transferLearningFinishedUpdateHandlerCallback(msg: Message): Boolean {
        return true
    }

    fun transferLearningSwitchCallback(): Boolean {
        // Transfer Learning was just enabled.
        if (transferLearningSwitch.isChecked) {
            val ret = doTransferLearning()
            if (ret < 0) {
                return false
            }
        } else {
            quitTransferLearning()
        }
        return true
    }

    fun printLearningLossHandlerCallback(msg: Message): Boolean {
        val msg_converted = msg.obj as Pair<Int, Float>
        print("\tEpoch %d: Loss = %f\n".format(msg_converted.first, msg_converted.second))
        lossTextView.text = resources.getString(R.string.loss_label_val, msg_converted.second)
        return true
    }

    fun debugButtonCallback(msg: Message): Boolean {
        debug()
        return true
    }

    fun setActivityProbabilities(mostProbableActivities: Array<Int>, values:Array<Array<Double>>) {

        val activity_tw_left: TextView = findViewById(R.id.classified_activity_left)
        val activity_tw_right: TextView = findViewById(R.id.classified_activity_right)

        // Display the most probable activity determined by both, pre-trained and transfer-learning model
        val text_left = resources.getString(R.string.classified_activity_left_val, activity_labels[mostProbableActivities[0]])
        val text_right = resources.getString(R.string.classified_activity_right_val, activity_labels[mostProbableActivities[1]])
        activity_tw_left.text = text_left
        activity_tw_right.text = text_right


        for (k in 0 until values[0].size) {
            val tw_id_left: Int = resources.getIdentifier("activity_%d_left".format(k+1), "id", packageName )
            val tw_left: TextView = findViewById(tw_id_left)

            val str_id_left = resources.getIdentifier("activity_%d_left_val".format(k+1), "string", packageName)
            tw_left.text = resources.getString(str_id_left, values[0][k])

            val tw_id_right: Int = resources.getIdentifier("activity_%d_right".format(k+1), "id", packageName )
            val tw_right: TextView = findViewById(tw_id_right)

            val str_id_right = resources.getIdentifier("activity_%d_right_val".format(k+1), "string", packageName)
            tw_right.text = resources.getString(str_id_right, values[1][k])
        }
    }

    fun setTransferLearningActivitySamplingStats(id:Int) {
        var N_samples: Int = -1
        if (id == 0) {
            N_samples = walkingTransferLearningSamples.size
        } else if (id == 1) {
            N_samples = walkingUpstairsTransferLearningSamples.size
        } else if (id == 2) {
            N_samples = walkingDownstairsTransferLearningSamples.size
        } else if (id == 3) {
            N_samples = runningTransferLearningSamples.size
        }

        val tw_id_center: Int = resources.getIdentifier("tfl_timestamps_%d_center".format(id + 1), "id", packageName)
        val tw_center: TextView = findViewById(tw_id_center)

        val tw_id_right: Int = resources.getIdentifier("tfl_timestamps_%d_right".format(id + 1), "id", packageName)
        val tw_right: TextView = findViewById(tw_id_right)

        val date = Calendar.getInstance().time
        tw_center.text = DateFormat.getTimeInstance(DateFormat.SHORT).format(date)
        tw_right.text = N_samples.toString()
    }

    fun getTransferLearningDataCapturingSwitchState(): Boolean {
        var state = false
        captureDataSwitchStateLock.lock()
        try {
            state = captureDataSwitchState
        } finally {
            captureDataSwitchStateLock.unlock()
        }
        return state
    }

    fun doTransferLearning(): Int {
        if (!transferLearningSampleCounts.all { element -> element > 0 }) {
            showAlert("You have to provide samples for all activities.", this)
            return - 1
        }

        print("Adding samples...\n")
        addTransferLearningSamplesToModel(walkingTransferLearningSamples.toTypedArray(), "WALKING")
        addTransferLearningSamplesToModel(walkingUpstairsTransferLearningSamples.toTypedArray(), "WALKING_UPSTAIRS")
        addTransferLearningSamplesToModel(walkingDownstairsTransferLearningSamples.toTypedArray(), "WALKING_DOWNSTAIRS")
        addTransferLearningSamplesToModel(runningTransferLearningSamples.toTypedArray(), "RUNNING")
        print("Starting training...\n")
        transferLearningModel.enableTraining { epoch, loss ->  printLearningLoss(epoch, loss)}

        return 0
    }
    
    fun quitTransferLearning() {
        transferLearningModel.disableTraining()
    }

    fun printLearningLoss(epoch:Int, loss:Float) {
        val msg = Message()
        msg.obj = Pair(epoch, loss)
        printLearningEpochHandler.sendMessage(msg)
    }

    fun addTransferLearningSamplesToModel(dataArray: Array<Array<Array<Double>>>, label: String) {
        val N_samples = dataArray.size
        val sample_count_start = transferLearningModel.model.sampleCount
        for (k in 0 until N_samples) {
            //print("\tAdding Sample %d of %d for activity %s\n".format(k, N_samples, label))
            val reshapedData = reshapeSampleData(dataArray[k])
            transferLearningModel.addSample(reshapedData, label)
            // Adding samples to the model is an asynchronous task. To reduce the possible sources
            // of runtime-issues, the samples are added in a synchronous way.
            while (transferLearningModel.model.sampleCount - sample_count_start < k+1) {
                Thread.sleep(10)
            }
        }

        print("Sample-count: %d, Added Samples: %d\n".format(N_samples, transferLearningModel.model.sampleCount))
    }

    fun reshapeSampleData(sampleData: Array<Array<Double>>): FloatArray {
        val N_samples: Int = sampleData[0].size
        val N_columns: Int = 6

        val out_array = FloatArray(N_samples * 6, init = {0.0F})
        for (k in 0 until N_samples) {
            for (l in 0 until N_columns) {
                val idx = k * N_columns + l
                out_array[idx] = sampleData[l][k].toFloat()
            }
        }
        return out_array
    }

    fun showAlert(message:String, context: Context, title: String="Alert") {
        val builder = AlertDialog.Builder(context)
        builder.setMessage(message)
        builder.setTitle("Alert")
        builder.setNeutralButton("Ok",
            DialogInterface.OnClickListener { dialog, id ->
                dialog.dismiss()
            })

        val dialog = builder.create()
        dialog.show()
    }

    fun showAppStateAlert() {
        showAlert(message = "Transfer Learning and Inference cannot be active at the same time.",
            context = this)
    }
    
    fun updateLossValue(loss:Double) {
        lossTextView.text = resources.getString(R.string.loss_label_val, loss)
    }

    override fun onClick(v: View?) {
        if (v?.id == bReturnToMainButton.id) {
            val intent = Intent(this, MainActivity::class.java);
            startActivity(intent);
        } else if (v?.id == bDebug.id) {
            val msg = Message()
            debugHandler.sendMessage(msg)
        }
    }

    // Callback for Spinner (AdapterView.OnItemSelectedListener)
    override fun onItemSelected(parent: AdapterView<*>, view: View?, pos: Int, id: Long) {

        // An item was selected. You can retrieve the selected item using
        // parent.getItemAtPosition(pos)
        val selected_item = parent.getItemAtPosition(pos)
        tflActivitySpinnerSelectedItemPos = pos
        tflActivitySpinnerSelectedItemString = selected_item.toString()
    }

    // Must be implemented by class for AdapterView.OnItemSelectedListener
    override fun onNothingSelected(parent: AdapterView<*>) {}

    // Listener for data-capturing switch state change.
    override fun onCheckedChanged(buttonView: CompoundButton?, isChecked: Boolean) {

        // State of data capturing switch changed
        if (buttonView?.id == captureDataSwitch.id) {
            // User attempts to enable different tasks at same time
            if (isChecked && (inferenceSwitchState || transferLearningSwitchState)) {
                showAppStateAlert()
                buttonView.isChecked = false
                return
            }

            // Thread-safe setting of button state
            captureDataSwitchStateLock.lock()
            try {
                captureDataSwitchState = isChecked
            } finally {
                captureDataSwitchStateLock.unlock()
            }

            // Trigger data capturing if possible
            if (transferLearningDataCapturingLock.tryLock()) {
                try {
                    transferLearningDataCapturingCondition.signal()
                } finally {
                    transferLearningDataCapturingLock.unlock()
                }
            }
            // State of inference button changed
        } else if (buttonView?.id == inferenceSwitch.id) {
            // User attempts to enable different tasks at same time
            if (isChecked && (captureDataSwitchState || transferLearningSwitchState)) {
                showAppStateAlert()
                buttonView.isChecked = false
                return
            }

            if (isChecked) {
                Thread.sleep(2000)
                classificationLock.unlock()
                print("Enabling classification")
            } else {
                classificationLock.lock()
                print("Disabling classification")
            }

            // No thread-safe setting required as all actions based on this state are done in the
            // UI-thread.
            inferenceSwitchState = isChecked
        } else if (buttonView?.id == transferLearningSwitch.id) {
            // User attempts to enable different tasks at same time
            if (isChecked && (inferenceSwitchState || captureDataSwitchState)) {
                showAppStateAlert()
                buttonView.isChecked = false
                return
            }

            val success = transferLearningSwitchCallback()
            if (!success) {
                transferLearningSwitch.isChecked = false
                transferLearningSwitchState = false
            } else {
                transferLearningSwitchState = isChecked
            }
        } else if (buttonView?.id == transferLearningCaptureModeSwitch.id) {
            transferLearningCaptureModeSwitchState = isChecked
        }
    }

    fun debug() {

        val rawSensorDataPath: String = "raw_sensor_data/"
        val activitySubfolders = arrayOf("walking_tf", "walking_upstairs_tf", "walking_downstairs_tf", "running_tf")
        val ids = arrayOf(0,1,2,3)

        for (k in ids.indices) {
            val assetSubdir = rawSensorDataPath + activitySubfolders[k] + "/"
            val filesInSubdir = assets.list(assetSubdir)

            val gyroFilesInSubdir = filesInSubdir?.filter { elem -> elem.contains("gyro") }
            val N_gyro_files: Int = gyroFilesInSubdir?.size ?: 0

            transferLearningCaptureModeSwitch.isChecked = true
            for (l in 0 until N_gyro_files) {
                val gyroFileName = gyroFilesInSubdir?.get(l)
                val accelFileName = gyroFileName?.replace("gyro", "accel")

                val gyro_raw_file_stream = assets.open(assetSubdir + gyroFileName)
                val accel_raw_file_stream = assets.open(assetSubdir + accelFileName)

                val gyroData = loadRawSensorDataFromCSV(gyro_raw_file_stream)
                val accelData = loadRawSensorDataFromCSV(accel_raw_file_stream)

                val activitySamples = tflDataCapturingThread.doProcessing(gyroData, accelData)
                // Finally, copy the derived samples to correct storage array
                tflDataCapturingThread.copyActivitySamples(activitySamples, ids[k])
                tflDataCapturingThread.updateUI(ids[k], activitySamples.size)
            }
            transferLearningCaptureModeSwitch.isChecked = false



        }

        /*
        val testSetData = TFLClassificationThread.loadSampleSet(
            path = "transfer_learning_device_dataset/training_set",
            //path = "transfer_learning_device_dataset/test_set",
            assetManager = this.assets)
        debugAddTrainingSamples(testSetData)

        val msg0 = Message()
        msg0.obj = Pair(0, transferLearningSampleCounts[0])
        transferLearningActivitySamplingFinishedHandler.sendMessage(msg0)

        val msg1 = Message()
        msg1.obj = Pair(1, transferLearningSampleCounts[1])
        transferLearningActivitySamplingFinishedHandler.sendMessage(msg1)

        val msg2 = Message()
        msg2.obj = Pair(2, transferLearningSampleCounts[2])
        transferLearningActivitySamplingFinishedHandler.sendMessage(msg2)

        val msg3 = Message()
        msg3.obj = Pair(3, transferLearningSampleCounts[3])
        transferLearningActivitySamplingFinishedHandler.sendMessage(msg3)
        */
    }

    fun debugAddTrainingSamples(samples: MutableList<TestSetSample>) {
        val labels = arrayOf("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "RUNNING")
        val N_samples = samples.size

        val walkingSamples = mutableListOf<Array<Array<Double>>>()
        val walkingUpstairsSamples = mutableListOf<Array<Array<Double>>>()
        val walkingDownstairsSamples = mutableListOf<Array<Array<Double>>>()
        val runningSamples = mutableListOf<Array<Array<Double>>>()
        for (k in 0 until N_samples) {
            val id = samples[k].label
            print("\tAdding Sample %d of %d for activity %s\n".format(k+1, N_samples, id))

            val data_sample = arrayOf(
                    samples[k].gyroSensorData[0],
                    samples[k].gyroSensorData[1],
                    samples[k].gyroSensorData[2],
                    samples[k].accelSensorData[0],
                    samples[k].accelSensorData[1],
                    samples[k].accelSensorData[2]
                )

            if (id == 0) {
                walkingSamples.add(data_sample)
            } else if (id == 1) {
                walkingUpstairsSamples.add(data_sample)
            } else if (id == 2) {
                walkingDownstairsSamples.add(data_sample)
            } else if (id == 3) {
                runningSamples.add(data_sample)
            }
        }

        // Capture-mode is set to "Overwrite"
        if (!transferLearningCaptureModeSwitchState) {
            walkingTransferLearningSamples.clear()
            walkingDownstairsTransferLearningSamples.clear()
            walkingUpstairsTransferLearningSamples.clear()
            runningTransferLearningSamples.clear()
        }

        walkingTransferLearningSamples.addAll(walkingSamples)
        walkingDownstairsTransferLearningSamples.addAll(walkingDownstairsSamples)
        walkingUpstairsTransferLearningSamples.addAll(walkingUpstairsSamples)
        runningTransferLearningSamples.addAll(runningSamples)

        transferLearningSampleCounts[0] = walkingTransferLearningSamples.size
        transferLearningSampleCounts[1] = walkingUpstairsTransferLearningSamples.size
        transferLearningSampleCounts[2] = walkingDownstairsTransferLearningSamples.size
        transferLearningSampleCounts[3] = runningTransferLearningSamples.size
    }

    fun loadRawSensorDataFromCSV(stream: InputStream): SensorData {
        // Read data in CSV-format from given input-stream
        val timestamps: MutableList<Long> = mutableListOf();
        val data: MutableList<Array<Double>> = mutableListOf();
        val reader = stream.bufferedReader()
        var line: String? = reader.readLine();
        while (line != null) {
            val tokens = line.split(";")
            try {
                val timestamp: Long = tokens[0].toLong()
                val sensorValues: Array<Double> =
                    arrayOf(
                        tokens[1].toDouble(), // X
                        tokens[2].toDouble(), // Y
                        tokens[3].toDouble()  // Z
                    )
                timestamps.add(element = timestamp)
                data.add(element = sensorValues);

            } catch (e: NumberFormatException) {}
            line = reader.readLine()
        }
        reader.close()

        /* Currently, the <data>-array is set up in a row-first index order, with the lowest
           time-value at row 0.
           To be compatible to the signal-processing flow, the structure has to be changed to
           column-first index-order with the highest time-value at row 0.
           The time-reversal have, of course, also to be applied to the timestamp-vector.
         */
        val N_samples = timestamps.size
        val dataOut: Array<Array<Double>> = arrayOf(
            Array<Double>(N_samples, init = {0.0}),
            Array<Double>(N_samples, init = {0.0}),
            Array<Double>(N_samples, init = {0.0})
        )

        for (k in 0 until N_samples) {
            dataOut[0][k] = data[N_samples - 1 - k][0]
            dataOut[1][k] = data[N_samples - 1 - k][1]
            dataOut[2][k] = data[N_samples - 1 - k][2]
        }

        val timestampsOut: Array<Long> =
            Array<Long>(size = N_samples, init = {idx -> timestamps[N_samples - 1 - idx]})

        return SensorData(timestampsOut, dataOut)
    }
}


class TFLClassificationThread
constructor(val activityThread: TransferLearning): Thread() {
    companion object {

        fun loadSampleSet(path: String, assetManager: AssetManager): MutableList<TestSetSample> {
            val sampleFiles = assetManager.list(path)
            val NTestsetFiles:Int = sampleFiles?.size ?: 0

            val data = mutableListOf<TestSetSample>()
            for (k in 0 until NTestsetFiles) {
                val sample_filename:String = sampleFiles?.get(k)?: ""
                val sample_stream = assetManager.open(path + "/%s".format(sample_filename))
                data.add(loadActivitySample(sample_stream, sample_filename))
            }
            return data
        }

        fun loadActivitySample(stream: InputStream, filename:String, delimiter:String=";"): TestSetSample {

            // Read data in CSV-format from given input-stream
            val data: MutableList<Array<Double>> = mutableListOf();
            val reader = stream.bufferedReader()
            var line: String? = reader.readLine();
            while (line != null) {
                val tokens = line.split(";")
                try {
                    val tmp = arrayOf<Double>(
                        tokens[0].toDouble(), // Gyro X
                        tokens[1].toDouble(), // Gyro Y
                        tokens[2].toDouble(), // Gyro Z
                        tokens[3].toDouble(), // Accel X
                        tokens[4].toDouble(), // Accel Y
                        tokens[5].toDouble()  // Accel Z
                    )
                    data.add(element = tmp);

                } catch (e: NumberFormatException) {}
                line = reader.readLine()
            }
            reader.close()

            val N_samples = data.size
            val dataArray: Array<Array<Double>> = arrayOf(
                Array<Double>(N_samples, init = {0.0}),
                Array<Double>(N_samples, init = {0.0}),
                Array<Double>(N_samples, init = {0.0}),
                Array<Double>(N_samples, init = {0.0}),
                Array<Double>(N_samples, init = {0.0}),
                Array<Double>(N_samples, init = {0.0})
            )

            // Convert from row-indices first to column-indices first, to match the required data-format
            for (k in 0 until N_samples) {
                dataArray[0][k] = data[k][0]
                dataArray[1][k] = data[k][1]
                dataArray[2][k] = data[k][2]
                dataArray[3][k] = data[k][3]
                dataArray[4][k] = data[k][4]
                dataArray[5][k] = data[k][5]
            }


            // Extract activity label from filename
            val re = "activity_[0-9]+".toRegex()
            val match = re.find(filename)?.value ?: ""
            val label_match = "[0-9]+".toRegex().find(match)?.value ?: "-1"
            val label = label_match.toInt()

            val gyroSensorData = arrayOf(
                dataArray[0],
                dataArray[1],
                dataArray[2])

            val accelSensorData = arrayOf(
                dataArray[3],
                dataArray[4],
                dataArray[5])


            return TestSetSample(gyroSensorData, accelSensorData, label, filename)
        }
    }

    val TEST:Boolean = false
    val TEST_TRANSFER_LEARNING = false
    lateinit var pretrainedModelTrainingSet: MutableList<TestSetSample>;
    var pretrainedModelTrainingSetCounter: Int = 0
    lateinit var transferLearningTrainingSet: MutableList<TestSetSample>;
    var transferLearningTrainingSetCounter: Int = 0

    val synchronizationLock = ReentrantLock()
    val synchronizationCondition = synchronizationLock.newCondition()
    val model_transfer_learning: TransferLearningModelWrapper =
        TransferLearningModelWrapper(activityThread)

    lateinit var pretrainedModel: ConvertedBaseModel;
    lateinit var pretrainedModelInputBuffer: TensorBuffer;
    lateinit var currentDebugSample: TestSetSample



    // Set this to true to terminate the thread.
    var terminateThread: Boolean = false;

    override fun run() {
        pretrainedModel = ConvertedBaseModel.newInstance(activityThread)
        pretrainedModelInputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 1200), DataType.FLOAT32)


        if (TEST) {
            val data = loadTrainingSets()
            pretrainedModelTrainingSet = data.first
            transferLearningTrainingSet = data.second

            pretrainedModelTrainingSetCounter = 0
            transferLearningTrainingSetCounter = 0
        }

        while (!terminateThread) {
            // Wait until the timer-task triggers the classification
            synchronizationLock.lock()
            synchronizationCondition.await()

            // Arrays filled with the sensor-data, either with runtime-data for testset-data
            var gyroData: SensorData;
            var accelData: SensorData;

            activityThread.writeDataLock.lock()
            try {
                gyroData = SignalProcessingUtilities.deepcopySensorDataBuffer(activityThread.gyroTimeArray, activityThread.gyroSensorArray)
                accelData = SignalProcessingUtilities.deepcopySensorDataBuffer(activityThread.accelTimeArray, activityThread.accelSensorArray)
            } finally {
                activityThread.writeDataLock.unlock()
            }

            /* Debugging code. Remove on Release*/
            //val gyro_raw_file_stream = activityThread.assets.open("raw_sensor_data/Running_gyro_sensor_data_1623697180289_original_clipped.csv")
            //val accel_raw_file_stream = activityThread.assets.open("raw_sensor_data/Running_accel_sensor_data_1623697180289_original_clipped.csv")

            //gyroData = activityThread.loadRawSensorDataFromCSV(gyro_raw_file_stream)
            //accelData = activityThread.loadRawSensorDataFromCSV(accel_raw_file_stream)


            val (activityArray, valueArray) = doComputationTFLite(gyroData, accelData)
            if (activityArray[0] >= 0 && activityArray[1] >= 0) {
                val msg = Message()
                msg.obj = Pair(activityArray, valueArray)
                activityThread.probabilityUpdateHandler.sendMessage(msg)
            }
        }
    }

    private fun doComputationTFLite(gyroData: SensorData,
                                    accelData: SensorData): Pair<Array<Int>,Array<Array<Double>>> {

        // Clip the time-series to an interval of 4 seconds. (Times are given in nanoseconds.)
        val T_clip: Long = 4_000_000_000
        val (gyroDataClipped, accelDataClipped) =
            clipData(gyroData, accelData, T_clip = T_clip);

        // Too data available
        if (gyroDataClipped.first.isEmpty() || accelDataClipped.first.isEmpty()) {
            return Pair(arrayOf(-1,-1), arrayOf(
                Array<Double>(size=3, init={0.0}),
                Array<Double>(size=3, init={0.0})))
        }


        val T_start_gyro: Long = gyroDataClipped.first[0]
        val T_end_gyro: Long = gyroDataClipped.first.last()
        val T_start_accel: Long = accelDataClipped.first[0]
        val T_end_accel: Long = accelDataClipped.first.last()

        // Captured Data-Frame to short
        if (T_end_gyro - T_start_gyro < T_clip ||
            T_end_accel - T_start_accel < T_clip) {
            // When too less data is available, return dummy values.
            return Pair(arrayOf(-1,-1), arrayOf(
                Array<Double>(size=3, init={0.0}),
                Array<Double>(size=3, init={0.0})))
        }

        val resampledSensorData = doResampling(
            gyroDataClipped,
            accelDataClipped,
            fs=50.0,
            Ns=200)

        val normalizedSensorData = doNormalization(
            resampledSensorData[0],
            resampledSensorData[1])

        var data: FloatArray;

        if (TEST) {
            if (TEST_TRANSFER_LEARNING) {
                //currentDebugSample = transferLearningTrainingSet[transferLearningTrainingSetCounter]
                currentDebugSample = transferLearningTrainingSet[transferLearningTrainingSetCounter]
            } else {
                currentDebugSample = pretrainedModelTrainingSet[pretrainedModelTrainingSetCounter]
            }

            data = reshapeDataForTensorFlow(
                currentDebugSample.gyroSensorData,
                currentDebugSample.accelSensorData)
        } else {
            data = reshapeDataForTensorFlow(
                normalizedSensorData[0],
                normalizedSensorData[1])
        }

        // Run the inference
        val (pretrained_model_out, transfer_learninig_model_out) = inference(data)

       val activityArray: Array<Int> = arrayOf(
           getActivityIdxWithHighestProbaility(pretrained_model_out),
           getActivityIdxWithHighestProbaility(transfer_learninig_model_out))

       val valueArray: Array<Array<Double>> = arrayOf(
           pretrained_model_out,
           transfer_learninig_model_out
       )
        if (TEST) {
            val expextedActivity = currentDebugSample.label
            val pretrained_model_activity = activityArray[0]
            val transfer_learninig_model_activity = activityArray[1]

            print("Expected Activity: %2d, Pretrained-model activity: %2d, Transfer-learning activity: %2d\n".format(expextedActivity, pretrained_model_activity, transfer_learninig_model_activity))
            if (TEST_TRANSFER_LEARNING) {
                transferLearningTrainingSetCounter =
                    (transferLearningTrainingSetCounter + 1) % transferLearningTrainingSet.size

            } else {
                pretrainedModelTrainingSetCounter =
                    (pretrainedModelTrainingSetCounter + 1) % pretrainedModelTrainingSet.size
            }
        }
        return Pair<Array<Int>, Array<Array<Double>>>(activityArray, valueArray)
    }

    private fun reshapeDataForTensorFlow(gyroSensorData: Array<Array<Double>>,
                                         accelSensorData: Array<Array<Double>>): FloatArray {

        val data2D: Array<Array<Double>> = arrayOf(
            gyroSensorData[0],
            gyroSensorData[1],
            gyroSensorData[2],
            accelSensorData[0],
            accelSensorData[1],
            accelSensorData[2]
        )

        return activityThread.reshapeSampleData(data2D)
    }

    private fun inference(data: FloatArray): Pair<Array<Double>, Array<Double>> {

        // Run the transfer learning model
        var model_transfer_learning_out = model_transfer_learning.predict(data)

        // Run the pretrained model
        pretrainedModelInputBuffer.loadArray(data)
        val pretrainedModelOut = pretrainedModel.process(pretrainedModelInputBuffer)

        // Get the probabilities from the transfer-learning model as a double-array
        val transferLearningModelOutArray =
            model_transfer_learning_out.map { elem -> elem.confidence.toDouble() }.toTypedArray()

        // Get the probabilities from the pre-trained model as a double-array
        val pretrainedModelOutArray = pretrainedModelOut.outputFeature0AsTensorBuffer.floatArray.map { elem -> elem.toDouble() }.toTypedArray()


        //print("Pre-Trained model:\n")
        //for (k in 0 until 4) {
        //    print("\t%f".format(pretrainedModelOutArray[k]))
        //}
        //print("\nTransfer-learninig model:\n")
        //for (k in 0 until 4) {
        //    print("\t%f\t%f\n".format(transferLearningModelOutArray[k], model_transfer_learning_out[k].confidence))
        //}
        //print("\n")

        return Pair(pretrainedModelOutArray, transferLearningModelOutArray)
    }

    private fun getActivityIdxWithHighestProbaility(probabilities: Array<Double>): Int {
        val N_values = probabilities.size
        val indices = (0 until N_values).toList().toIntArray()
        val sortedIndices = indices.sortedByDescending { idx -> probabilities[idx] }
        return sortedIndices[0]
    }

    private fun doResampling(gyroData: SensorData,
                             accelData: SensorData,
                             fs:Double=100.0,
                             Ns:Int = -1): Array<Array<Array<Double>>> {
        /* The clipped data-series passed to this function are resampled to the given sampling
           frequency. As one cannot ensure that the two series are aligned in time, the higher of
           the two is used as reference-time passed to the resampling functions.
           This means that the interval BELOW T_align is omitted in the resampling.
        * */
        val T_gyro_low: Long = gyroData.first[0]
        val T_accel_low: Long = accelData.first[0]
        val T_align: Long;

        if (T_gyro_low > T_accel_low) {
            T_align = T_gyro_low
        } else {
            T_align = T_accel_low
        }

        val gyroSensorDataResampled =
            SignalProcessingUtilities.resampleTimeSeries(
                data = gyroData,
                fs=fs,
                N_samples_out = Ns,
                T_ref = T_align)

        val accelSensorDataResampled =
            SignalProcessingUtilities.resampleTimeSeries(
                data = accelData,
                fs=fs,
                N_samples_out = Ns,
                T_ref = T_align)

        return arrayOf(gyroSensorDataResampled, accelSensorDataResampled)
    }

    private fun doNormalization(gyroSensorData: Array<Array<Double>>,
                                accelSensorData: Array<Array<Double>>): Array<Array<Array<Double>>> {

        val gyroNormalized = SignalProcessingUtilities.sensorDataNormalization(gyroSensorData)
        val accelNormalized = SignalProcessingUtilities.sensorDataNormalization(accelSensorData)

        return arrayOf(gyroNormalized, accelNormalized)
    }

    // Clip both data-series to a given time-interval <T_clip>.
    // <T_clip> is given in integer nanoseconds.
    private fun clipData(
        gyroData: SensorData,
        accelData: SensorData,
        T_clip:Long=10_000_000_000): Pair<SensorData, SensorData> {

        // Sensor-Data capturing is event-based, so no uniformly spaced sample timestamps can be
        // assumed.
        // Therefore, the time-series from Gyroscope and Accelerometer have to be aligned somehow.
        // Here, the alignment is done according to the lower one of the highest timestamps for
        // Gyroscope and Accelerometer.
        // The clipping-functions below then clip the two data-series in a way that the samples
        // right next to the determined clipping boundary are contained in the returned
        // data-structures.
        // Therefore, the returned time-series always contain a slightly longer time-interval than
        // specified.
        val T_high_gyro: Long = gyroData.first[0]
        val T_high_accel: Long = accelData.first[0]
        var T_clip_high: Long = -1

        if (T_high_gyro > T_high_accel) {
            T_clip_high = T_high_accel
        } else {
            T_clip_high = T_high_gyro
        }

        val clippedGyroData =
            SignalProcessingUtilities.clipAndReverseDataSeries(
                data = gyroData,
                T_clip = T_clip,
                T_clip_high = T_clip_high);

        val clippedAccelData =
            SignalProcessingUtilities.clipAndReverseDataSeries(
                accelData,
                T_clip = T_clip,
                T_clip_high = T_clip_high);

        return Pair(clippedGyroData, clippedAccelData)
    }

    private fun loadTrainingSets(): Pair<MutableList<TestSetSample>, MutableList<TestSetSample>> {
        val datasetPretrained = loadSampleSet(
            path = "pretrained_model_device_dataset/test_set",
            assetManager = activityThread.assets)

        val datasetTransferLearning = loadSampleSet(
            path = "transfer_learning_device_dataset/test_set",
            assetManager = activityThread.assets)

        return Pair(datasetPretrained, datasetTransferLearning)
    }

}

class TFLDataCapturingThread
constructor(val activityThread: TransferLearning): Thread() {
    override fun run() {
        var currentActivityIndex: Int = 0

        while(true) {
            // Wait until the user starts capturing the activity
            waitUntilDataCaptureEnabled()
            // Get the index of the currently selected activity to be captured
            currentActivityIndex = activityThread.tflActivitySpinnerSelectedItemPos
            // Clean up the sensor-data buffers
            flushSensorDataBuffers()
            // Wait until the user has finished the activity
            waitUntilDataCaptureDisabled()

            // When locked, toggling the UI-button has not effect, except on its state-variable in the
            // activity-thread
            activityThread.transferLearningDataCapturingLock.lock()
            try {

                //val gyro_raw_file_stream = activityThread.assets.open("raw_sensor_data/Running_gyro_sensor_data_1623696041976.txt")
                //val accel_raw_file_stream = activityThread.assets.open("raw_sensor_data/Running_accel_sensor_data_1623696041976.txt")

                //val gyroData = activityThread.loadRawSensorDataFromCSV(gyro_raw_file_stream)
                //val accelData = activityThread.loadRawSensorDataFromCSV(accel_raw_file_stream)

                // Deepcopy the current values from the sensor-data buffers in a thread-save way
                val (gyroData, accelData) = copyBuffers()
                val activitySamples = doProcessing(gyroData, accelData)
                if (activitySamples.isEmpty()) {
                    return
                }
                // Finally, copy the derived samples to correct storage array
                copyActivitySamples(activitySamples, currentActivityIndex)
                updateUI(currentActivityIndex, activitySamples.size)

            } finally {
                activityThread.transferLearningDataCapturingLock.unlock()
            }
        }
    }

    fun waitUntilDataCaptureEnabled() {
        // Wait for the user to enable data capturing via the corresponding UI-switch
        while(!activityThread.getTransferLearningDataCapturingSwitchState()) {
            activityThread.transferLearningDataCapturingLock.lock()
            try {
                activityThread.transferLearningDataCapturingCondition.await()
            } finally {
                activityThread.transferLearningDataCapturingLock.unlock()
            }
        }
    }

    fun waitUntilDataCaptureDisabled() {
        // Wait for the user to disable data capturing via the corresponding UI-switch
        while(activityThread.getTransferLearningDataCapturingSwitchState()) {
            activityThread.transferLearningDataCapturingLock.lock()
            try {
                activityThread.transferLearningDataCapturingCondition.await()
            } finally {
                activityThread.transferLearningDataCapturingLock.unlock()
            }
        }
    }

    fun flushSensorDataBuffers() {
        // Flush the sensor data buffers
        activityThread.writeDataLock.lock()
        try {
            activityThread.flushSensorDataBuffers()
        } finally {
            activityThread.writeDataLock.unlock()
        }
    }

    fun copyBuffers(): Pair<SensorData, SensorData> {
        // Copy the sensor-data arrays
        var gyroData: SensorData;
        var accelData: SensorData;

        // Interrupt writing new sensor-data to the buffers
        activityThread.writeDataLock.lock()
        try {
            gyroData = SignalProcessingUtilities.deepcopySensorDataBuffer(
                activityThread.gyroTimeArray, activityThread.gyroSensorArray);

            accelData = SignalProcessingUtilities.deepcopySensorDataBuffer(
                activityThread.accelTimeArray, activityThread.accelSensorArray);
        } finally {
            activityThread.writeDataLock.unlock()
        }

        return Pair(gyroData, accelData)
    }

    fun doProcessing(gyroData: SensorData, accelData: SensorData): Array<Array<Array<Double>>> {

        val (gyroDataClipped, accelDataClipped) = clipData(gyroData, accelData)

        // The clipping function returns an empty time-array if the currently available timeframe
        // is to short
        if (gyroDataClipped.first.size == 0) {
            return Array<Array<Array<Double>>>(size=0, init={ Array<Array<Double>>(size = 0, init={Array<Double>(size = 0, init = { Double.NaN})})})
        }

        val (gyroDataResampled, accelDataResampled) = resampleData(gyroDataClipped, accelDataClipped)
        val timeframes = splitIntoTimeframes(gyroDataResampled, accelDataResampled)
        val normalizedTimeframes = doNormalization(timeframes)

        return normalizedTimeframes
    }

    fun clipData(gyroData: SensorData, accelData: SensorData): Pair<SensorData, SensorData> {
        val gyroDataClipped = SignalProcessingUtilities.clipAndReverseDataSeriesBackoff(gyroData, T_backoff = 5_000_000_000)
        val accelDataClipped = SignalProcessingUtilities.clipAndReverseDataSeriesBackoff(accelData, T_backoff = 5_000_000_000)

        return Pair(gyroDataClipped, accelDataClipped)
    }

    fun resampleData(gyroData: SensorData, accelData: SensorData):
            Pair<Array<Array<Double>>, Array<Array<Double>>> {

        val T_gyro_low: Long = gyroData.first[0]
        val T_accel_low: Long = accelData.first[0]
        val T_align: Long;

        if (T_gyro_low > T_accel_low) {
            T_align = T_gyro_low
        } else {
            T_align = T_accel_low
        }

        val gyroDataResampled =
            SignalProcessingUtilities.resampleTimeSeries(
                data = gyroData,
                fs = 50.0,
                T_ref = T_align)
        val accelDataResampled =
            SignalProcessingUtilities.resampleTimeSeries(
                data = accelData,
                fs = 50.0,
                T_ref = T_align)

        return Pair(gyroDataResampled, accelDataResampled)
    }

    fun splitIntoTimeframes(gyroSensorData: Array<Array<Double>>,
                            accelSensorData: Array<Array<Double>>,
                            N_samples_frame:Int = 200, fs:Double=50.0,
                            T_overlap:Double=3.0): Array<Array<Array<Double>>> {
        val N_samples_data: Int = gyroSensorData[0].size
        val N_samples_overlap: Int = (T_overlap * fs).toInt()
        val N_samples_non_overlap: Int = N_samples_frame - N_samples_overlap

        // Number of overlapping timeframes in data array
        val N_frames = 1 + ((N_samples_data -  N_samples_frame) / N_samples_non_overlap.toDouble()).toInt()

        val sampleList = mutableListOf<Array<Array<Double>>>()
        for (k in 0 until N_frames - 1) {
            val idx_start = k * N_samples_non_overlap
            val idx_end = idx_start + N_samples_frame - 1

            sampleList.add(arrayOf(
                gyroSensorData[0].slice(idx_start..idx_end).toTypedArray(),
                gyroSensorData[1].slice(idx_start..idx_end).toTypedArray(),
                gyroSensorData[2].slice(idx_start..idx_end).toTypedArray(),
                accelSensorData[0].slice(idx_start..idx_end).toTypedArray(),
                accelSensorData[1].slice(idx_start..idx_end).toTypedArray(),
                accelSensorData[2].slice(idx_start..idx_end).toTypedArray()
            ))
        }
        return sampleList.toTypedArray()
    }

    fun copyActivitySamples(samples: Array<Array<Array<Double>>>, activityID: Int) {
        if (activityID == 0) {
            // Switch is set to "Overwrite"
            if (!activityThread.transferLearningCaptureModeSwitchState) {
                activityThread.walkingTransferLearningSamples.clear()
            }
            // Actual copy
            activityThread.walkingTransferLearningSamples.addAll(samples)
            // Update sample-count
            activityThread.transferLearningSampleCounts[activityID] =
                activityThread.walkingTransferLearningSamples.size

        } else if (activityID == 1) {
            if (!activityThread.transferLearningCaptureModeSwitchState) {
                activityThread.walkingUpstairsTransferLearningSamples.clear()
            }
            activityThread.walkingUpstairsTransferLearningSamples.addAll(samples)
            activityThread.transferLearningSampleCounts[activityID] =
                activityThread.walkingUpstairsTransferLearningSamples.size

        } else if (activityID == 2) {
            if (!activityThread.transferLearningCaptureModeSwitchState) {
                activityThread.walkingDownstairsTransferLearningSamples.clear()
            }
            activityThread.walkingDownstairsTransferLearningSamples.addAll(samples)
            activityThread.transferLearningSampleCounts[activityID] =
                activityThread.walkingDownstairsTransferLearningSamples.size

        } else if (activityID == 3) {
            if (!activityThread.transferLearningCaptureModeSwitchState) {
                activityThread.runningTransferLearningSamples.clear()
            }
            activityThread.runningTransferLearningSamples.addAll(samples)
            activityThread.transferLearningSampleCounts[activityID] =
                activityThread.runningTransferLearningSamples.size
        }
    }

    fun updateUI(activityID:Int, N_samples:Int) {
        val msg = Message()
        msg.obj = Pair(activityID, N_samples)
        activityThread.transferLearningActivitySamplingFinishedHandler.sendMessage(msg)
    }

    fun doNormalization(timeframes: Array<Array<Array<Double>>>): Array<Array<Array<Double>>> {
        val N_timeframes = timeframes.size
        val N_samples = timeframes[0][0].size

        val normalizedTimeframes = Array<Array<Array<Double>>>(size=N_timeframes, init={
            Array(size = 6, init = {Array(size = N_samples, init = {0.0})})})

        for (k in 0 until N_timeframes) {
            normalizedTimeframes[k] = SignalProcessingUtilities.sensorDataNormalization(timeframes[k]).clone()
        }
        return normalizedTimeframes
    }
}

class TFLClassificationTimerTask
constructor(val classificationObject: TFLClassificationThread, val activityThread: TransferLearning): TimerTask() {

    override fun run() {

        // The activity's main-thread acquires the lock below to lock the classification process.
        if (activityThread.classificationLock.tryLock()) {
            // Trigger the classification
            classificationObject.synchronizationLock.lock()
            try {
                classificationObject.synchronizationCondition.signal()
            } finally {
                classificationObject.synchronizationLock.unlock()
                activityThread.classificationLock.unlock()
            }
        } else {
            //print("Cannot lock...\n")
        }
    }
}

class TestSetSample
    constructor(val gyroSensorData: Array<Array<Double>>,
                val accelSensorData: Array<Array<Double>>,
                val label:Int,
                val filename: String) {
}

class SignalProcessingUtilities {
    companion object {
        // Clip and time-reverse a data series to a certain period, optionally stating the upper time-boundary.
        // All times are stated in nanoseconds.
        fun clipAndReverseDataSeries(
            data: SensorData,
            T_clip: Long = 10_000_000_000, // Default clipping-time: 10 seconds
            T_clip_high: Long = -1): SensorData {

            val timeArray = data.first
            val dataArray = data.second

            // If the upper clipping-time is not given by the user, use the first entry of the
            // passed time-vector. (=the absolute highest time-value)
            var T_high: Long; // Contains the upper clipping in ns
            if (T_clip_high < 0) {
                T_high = timeArray[0]
            } else {
                T_high = T_clip_high
            }

            val N_samples = timeArray.size
            val T_low = T_high - T_clip

            // The first value <= T_low is searched. Remember that the array contains
            // the samples in descending order. (--> The firs row contains the latest sample.)
            var idx_T_start: Int = -1
            for (k in 0 until N_samples) {
                // Reached a negative time-value --> Too less data in buffer.
                if (timeArray[k] < 0) {
                    idx_T_start = k - 1
                    break;
                }
                if (timeArray[k] < T_low) {
                    idx_T_start = k
                    break;
                }
            }

            // Reverse the row-order to have the lowest time-value first.
            val dataArrayOut = arrayOf(
                dataArray[0].slice(idx_T_start downTo 0).toTypedArray(),
                dataArray[1].slice(idx_T_start downTo 0).toTypedArray(),
                dataArray[2].slice(idx_T_start downTo 0).toTypedArray())
            // Of course, also reverse the time-array
            val timeArrayOut = timeArray.slice(idx_T_start downTo 0).toTypedArray()

            return Pair(timeArrayOut, dataArrayOut);
        }

        // Clip and time-reverse a data-series with a given time-period removed at start and end.
        fun clipAndReverseDataSeriesBackoff(data: SensorData, T_backoff:Long=0): SensorData {
            val timeArray = data.first;
            val dataArray = data.second;

            val T_max = timeArray[0]

            // If the array is completely filled, T_lim contains a value >= 0. If not, its value is
            // <=0. In the latter case, the lowest time-value in the array is searched by iterating
            // over the timesteps.
            var T_min: Long = timeArray.last()
            // If T_min < 0 (the last row in the buffer is empty), the variable below holds the
            // index of the lowest time-value in the buffer.
            // If T_min >= 0, it simply holds the index of the last element in the buffer.
            var T_min_idx: Int = -1

            // The upper clipping boundary and the corresponding index
            val T_high: Long = T_max - T_backoff
            var T_high_idx:Int = -1

            // The lower clipping boundary and the corresponding index
            var T_low: Long = -1
            var T_low_idx:Int = -1

            val N_samples = dataArray[0].size

            // In case an error occurs during processing, the data-structure below is returned.
            val nonSuccessOutputArray = SensorData(
                Array<Long>(0, init = {-1}),
                arrayOf(
                    Array<Double>(0, init = {Double.NaN}),
                    Array<Double>(0, init = {Double.NaN}),
                    Array<Double>(0, init = {Double.NaN}))
            )


            // Search for the first time-value < T_high.
            for (k in 0 until N_samples) {
                // A negative time-value implies that the current and all upcoming rows are unfilled.
                // --> The array contains too less data. Return empty arrays instead.
                if (timeArray[k] < 0) {
                    return nonSuccessOutputArray
                }
                // Store the index of the timestamp right above T_high
                if (timeArray[k] < T_high) {
                    T_high_idx = k - 1
                    break;
                }
            }

            // Search for the lowest time-value in the buffer if it is not completely filled
            if (T_min < 0) {
                for (k in T_high_idx until N_samples) {
                    if (timeArray[k] < 0) {
                        T_min_idx = k - 1;
                        T_min = timeArray[T_min_idx]
                        break;
                    }
                }
            } else {
                T_min_idx = timeArray.lastIndex
            }

            // If the time-interval contained in the array is large enough, find the indices oof
            // the corresponding upper and lower boundaries. Otherwise, return an empty array.
            if (T_max - T_min < 2 * T_backoff) {
                return nonSuccessOutputArray
            }

            T_low = T_min + T_backoff
            // Iterate from the minimum time-value in direction of increasing values to find the
            // first time-value above the T_low boundary.
            for (k in T_min_idx downTo T_high_idx) {
                if (timeArray[k] >= T_low) {
                    T_low_idx = k + 1
                    break;
                }
            }

            // Reverse the row-order to have the lowest time-value first.
            val dataArrayOut = arrayOf(
                dataArray[0].slice(T_low_idx downTo T_high_idx).toTypedArray(),
                dataArray[1].slice(T_low_idx downTo T_high_idx).toTypedArray(),
                dataArray[2].slice(T_low_idx downTo T_high_idx).toTypedArray())
            // Of course, also reverse the time-array
            val timeArrayOut = timeArray.slice(T_low_idx downTo T_high_idx).toTypedArray()

            return SensorData(timeArrayOut, dataArrayOut)
        }

        fun resampleTimeSeries(data: SensorData,
                               fs:Double=100.0,
                               N_samples_out:Int = -1,
                               T_ref:Long = -1): Array<Array<Double>> {
            val timeArray = data.first // Remember: All times are given as integer nanoseconds
            val dataArray = data.second

            // The user can give a reference-time at which the resampling should start.
            // If this reference-time is smaller than the lowest time-value in the time-array,
            // use this lowest value.
            val t0: Long;
            if (T_ref >= 0L && T_ref >= timeArray[0]) {
                t0 = T_ref
            } else {
                t0 = timeArray[0]
            }

            val t_new = timeArray.map { tk -> tk - t0 };

            // If the number of output samples is not given by the user, determine it from
            // the available time-span and sampling-frequency.
            var __N_samples_out: Int = -1
            if (N_samples_out > 0) {
                __N_samples_out = N_samples_out
            } else {
                __N_samples_out = ((timeArray.last() - t0) * 1e-9 * fs ).toInt();
            }

            val x1 = Array<Double>(size=__N_samples_out, init={0.0});
            val x2 = Array<Double>(size=__N_samples_out, init={0.0});
            val x3 = Array<Double>(size=__N_samples_out, init={0.0});

            // Indices of the samples used for linear interpolation
            var left_sample_idx: Int = 0;
            var right_sample_idx: Int = 1;

            for (k: Int in 0 until __N_samples_out - 1) {
                // Time-value for sample k in integer nanoseconds.
                val tk: Long = ((k.toLong() * 1_000_000_000) / fs).toLong();

                // Find the first time-sample > tk.
                while (tk > t_new[right_sample_idx]) {
                     right_sample_idx++;
                }
                left_sample_idx = right_sample_idx - 1;

                // Remember: Times in nanoseconds!
                val t_left = t_new[left_sample_idx];
                val t_right = t_new[right_sample_idx];

                val x1_left = dataArray[0][left_sample_idx];
                val x1_right = dataArray[0][right_sample_idx];

                val x2_left = dataArray[1][left_sample_idx];
                val x2_right = dataArray[1][right_sample_idx];

                val x3_left = dataArray[2][left_sample_idx];
                val x3_right = dataArray[2][right_sample_idx];

                /* ========== Finally, the linear interpolation ========== */
                val slope_1 = (x1_right - x1_left) / (t_right - t_left);
                val slope_2 = (x2_right - x2_left) / (t_right - t_left);
                val slope_3 = (x3_right - x3_left) / (t_right - t_left);

                val delta_t = tk - t_left // In seconds

                x1[k] = slope_1 * delta_t + x1_left
                x2[k] = slope_2 * delta_t + x2_left
                x3[k] = slope_3 * delta_t + x3_left
            }

            return arrayOf(x1, x2, x3);
        }

        fun deepcopySensorDataBuffer(timeBuffer: Array<Long>,
                                     sensorBuffer: Array<Array<Double>>): SensorData {
            val arraySize = timeBuffer.size
            val timeArrayOut: Array<Long> = Array(size=arraySize, init = {idx -> timeBuffer[idx]})
            val sensorBufferOut: Array<Array<Double>> = arrayOf(
                Array(size = arraySize, init = {idx -> sensorBuffer[0][idx]}),
                Array(size = arraySize, init = {idx -> sensorBuffer[1][idx]}),
                Array(size = arraySize, init = {idx -> sensorBuffer[2][idx]})
            )

            return SensorData(timeArrayOut, sensorBufferOut)
        }

        fun sensorDataNormalization(buffer: Array<Array<Double>>): Array<Array<Double>> {

            val N_channels = buffer.size
            val N_samples = buffer[0].size

            val normalizedBuffer: Array<Array<Double>> =
                Array<Array<Double>>(size=N_channels, init={Array<Double>(size=N_samples, init={0.0})})

            for (k in 0 until N_channels) {
                val bufferAbs = buffer[k].map { elem -> elem.absoluteValue }
                val valMax = bufferAbs.maxOrNull() ?: Double.MAX_VALUE
                lateinit var normalizedChannel: Array<Double>;
                // If the data contained in the sensor-channel has a low maximum magnitude, fill
                // the corresponding output-channel with all zeros.
                if (valMax.absoluteValue < 1e-6) {
                    normalizedChannel = buffer[k].map { elem ->  0.0 }.toTypedArray()
                } else {
                    normalizedChannel = buffer[k].map { elem ->  elem / valMax }.toTypedArray()
                }

                normalizedBuffer[k] = normalizedChannel.clone()
            }
            print("\n")

            return normalizedBuffer
        }
    }
}
