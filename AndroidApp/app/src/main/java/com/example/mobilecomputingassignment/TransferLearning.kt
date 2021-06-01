package com.example.mobilecomputingassignment

import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.Message
import android.view.View
import android.widget.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel
import org.tensorflow.lite.examples.transfer.api.AssetModelLoader
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.LossConsumer
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.Prediction
import com.example.mobilecomputingassignment.ml.ConvertedModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.w3c.dom.Text
import java.io.InputStream
import java.lang.NumberFormatException
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.util.*
import java.util.concurrent.locks.ReentrantLock


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
    private lateinit var spActivitySelectionSpinner: Spinner;

    private lateinit var captureDataSwitch: Switch;
    private var captureDataSwitchState:Boolean = false;
    private var captureDataSwitchStateLock = ReentrantLock()

    /* ========== Sensor-data arrays ========== */

    // Assuming a sensor event occurring approximately every 150 ms, the buffer below is able to
    // hold approximately 150s of data for each activity. This should be far enough for both,
    // classification and transfer learning.
    val arraySize:Int = 1000
    val gyroSensorArray = arrayOf(
        Array<Double>(size=arraySize, init={-1.0}),
        Array<Double>(size=arraySize, init={0.0}),
        Array<Double>(size=arraySize, init={0.0}),
        Array<Double>(size=arraySize, init={0.0})
    );

    val accelSensorArray = arrayOf(
        Array<Double>(size=arraySize, init={-1.0}),
        Array<Double>(size=arraySize, init={0.0}),
        Array<Double>(size=arraySize, init={0.0}),
        Array<Double>(size=arraySize, init={0.0})
    );


    lateinit var probabilityUpdateHandler: Handler;

    val activity_labels: Array<String> = arrayOf(
        "Walking",
        "Walking Upstairs",
        "Walking Downstairs",
        "Sitting",
        "Standing",
        "Laying",
        "Stand to Sit",
        "Sit to Stand",
        "Sit to Lie",
        "Lie to Sit",
        "Stand to Lie",
        "Lie to Stand"
    )

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

        bReturnToMainButton = findViewById(R.id.button_transfer_learning_return);
        bReturnToMainButton.setOnClickListener(this)

        captureDataSwitch = findViewById(R.id.tfl_recording_switch)
        captureDataSwitch.setOnCheckedChangeListener(this)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager;
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        classificationThread = TFLClassificationThread(this);
        classificationTimer = Timer();
        classificationTimerTask = TFLClassificationTimerTask(classificationThread, this);

        classificationThread.start()

        tflDataCapturingThread = TFLDataCapturingThread(this)
        tflDataCapturingThread.start()


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
            // Get the timestamp in seconds
            val time: Double = event.timestamp.toDouble() * 1e-9F;

            writeDataLock.lock()
            try {
                pushSensorValuesToBuffer(gyroSensorArray, arrayOf(time, gyroX, gyroY, gyroZ))

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
                pushSensorValuesToBuffer(accelSensorArray, arrayOf(time, accelX, accelY, accelZ))
            } finally {
                writeDataLock.unlock()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Mandatory implementation by child-class of SensorEventListener()
    }

    private fun shiftRightBufferEntries(buffer: Array<Array<Double>>) {
        val N_columns: Int = buffer.size
        val N_entries: Int = buffer[0].size


        // Get the number of non-empty rows
        var N_filled_entries: Int = 0
        for (k in 0 until N_entries) {
            if (buffer[0][k] < 0) {
                N_filled_entries = k
                break;
            }
        }

        for (k in N_filled_entries downTo 1) {
            for (l in 0 until N_columns) {
                buffer[l][k] = buffer[l][k - 1]
            }
        }

        // Finally, fill the current row with NANs
        for (l in 0 until N_columns) {
            buffer[l][0] = Double.NaN
        }
    }

    private fun pushSensorValuesToBuffer(buffer: Array<Array<Double>>, values: Array<Double>) {
        shiftRightBufferEntries(buffer)
        for (k in 0 until values.size) {
            buffer[k][0] = values[k];
        }
    }

    // This method is NOT thread-safe!
    fun flushSensorDataBuffers() {

        gyroSensorArray[0].fill(-1.0)
        gyroSensorArray[1].fill(0.0)
        gyroSensorArray[2].fill(0.0)
        gyroSensorArray[3].fill(0.0)

        accelSensorArray[0].fill(-1.0)
        accelSensorArray[1].fill(0.0)
        accelSensorArray[2].fill(0.0)
        accelSensorArray[3].fill(0.0)
    }

    fun probabilityUpdateHandlerCallback(msg: Message):Boolean {
        val msg_converted = msg.obj as Pair<Array<Int>, Array<Array<Double>>>
        setActivityProbabilities(msg_converted.first, msg_converted.second)
        return true
    }

    fun setActivityProbabilities(activities: Array<Int>, values:Array<Array<Double>>) {

        val activity_tw_left: TextView = findViewById(R.id.classified_activity_left)
        val activity_tw_right: TextView = findViewById(R.id.classified_activity_right)

        activity_tw_left.text = resources.getString(R.string.classified_activity_left_val, activity_labels[activities[0]])
        activity_tw_right.text = resources.getString(R.string.classified_activity_right_val, activity_labels[activities[1]])


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
    override fun onClick(v: View?) {
        if (v?.id == this.bReturnToMainButton.id) {
            val intent = Intent(this, MainActivity::class.java);
            startActivity(intent);
        }
    }

    // Callback for Spinner (AdapterView.OnItemSelectedListener)
    override fun onItemSelected(parent: AdapterView<*>, view: View?, pos: Int, id: Long) {

        // An item was selected. You can retrieve the selected item using
        // parent.getItemAtPosition(pos)
        val selected_item = parent.getItemAtPosition(pos)
    }

    // Must be implemented by class for AdapterView.OnItemSelectedListener
    override fun onNothingSelected(parent: AdapterView<*>) {}

    // Listener for data-capturing switch state change.
    override fun onCheckedChanged(buttonView: CompoundButton?, isChecked: Boolean) {

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
    }
}


class TFLClassificationThread
constructor(val activityThread: TransferLearning): Thread() {

    val TEST:Boolean = false
    lateinit var testSetData: MutableList<TestSetSample>;
    var testsetSampleCounter: Int = 0

    val synchronizationLock = ReentrantLock()
    val synchronizationCondition = synchronizationLock.newCondition()
    val model_transfer_learning: TransferLearningModelWrapper =
        TransferLearningModelWrapper(activityThread)

    lateinit var pretrainedModel: ConvertedModel;
    lateinit var pretrainedModelInputBuffer: TensorBuffer;



    // Set this to true to terminate the thread.
    var terminateThread: Boolean = false;
    private val sampling_frequency: Double = 100.0;

    override fun run() {
        pretrainedModel = ConvertedModel.newInstance(activityThread)
        pretrainedModelInputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 1200), DataType.FLOAT32)


        if (TEST) {
            print("Loading testset...")
            testSetData = loadTestSet()
            testsetSampleCounter = 0
            print("Done\n")
        }

        while (!terminateThread) {


            // Wait until the timer-task triggers the classification
            synchronizationLock.lock()
            synchronizationCondition.await()

            // Arrays filled with the sensor-data, either with runtime-data for testset-data
            var gyroSensorData: Array<Array<Double>>;
            var accelSensorData: Array<Array<Double>>;

            activityThread.writeDataLock.lock()
            try {
                gyroSensorData = SignalProcessingUtilities.deepcopySensorDataBuffer(activityThread.gyroSensorArray);
                accelSensorData = SignalProcessingUtilities.deepcopySensorDataBuffer(activityThread.accelSensorArray);
            } finally {
                activityThread.writeDataLock.unlock()
            }

            val (activityArray, valueArray) = doComputationTFLite(gyroSensorData, accelSensorData)
            if (activityArray[0] >= 0 && activityArray[1] >= 0) {
                val msg = Message()
                msg.obj = Pair(activityArray, valueArray)
                activityThread.probabilityUpdateHandler.sendMessage(msg)

            }
        }
    }

    private fun doComputationTFLite(gyroSensorData: Array<Array<Double>>,
                                    accelSensorData: Array<Array<Double>>): Pair<Array<Int>,Array<Array<Double>>> {

        //Extract the available data from the buffer it is not filled completely
        val clippedData = clipData(gyroSensorData, accelSensorData, T_lim = 4.0);

        // Too less data available
        if (clippedData[0][0].last() - clippedData[0][0].first() < 4.0 || clippedData[1][0].last() - clippedData[1][0].first() < 4.0) {
            return Pair(arrayOf(-1,-1), arrayOf(
                Array<Double>(size=12, init={0.0}),
                Array<Double>(size=12, init={0.0})))
        }

        val resampledSensorData = doResampling(
            clippedData[0],
            clippedData[1],
            fs=50.0,
            Ns=200)

        val normalizedSensorData = doNormalization(
            resampledSensorData[0],
            resampledSensorData[1])

        var data: FloatArray;

        if (TEST) {
            data = reshapeDataForTensorFlow(
                testSetData[testsetSampleCounter].gyroSensorData,
                testSetData[testsetSampleCounter].accelSensorData)



        } else {
            data = reshapeDataForTensorFlow(
                normalizedSensorData[0],
                normalizedSensorData[1])
        }

        val (pretrained_model_out, transfer_learninig_model_out) = inference(data)

       val activityArray: Array<Int> = arrayOf(
           getActivityIdxWithHighestProbaility(pretrained_model_out),
           getActivityIdxWithHighestProbaility(transfer_learninig_model_out))

       val valueArray: Array<Array<Double>> = arrayOf(
           pretrained_model_out,
           transfer_learninig_model_out
       )
        if (TEST) {
            val expextedActivity = testSetData[testsetSampleCounter].label
            val pretrained_model_activity = activityArray[0]
            val transfer_learninig_model_activity = activityArray[1]

            print("Expected Activity: %2d, Pretrained-model activity: %2d, Transfer-learning activity: %2d\n".format(expextedActivity, pretrained_model_activity, transfer_learninig_model_activity))

            testsetSampleCounter = (testsetSampleCounter + 1) % testSetData.size
        }
        return Pair<Array<Int>, Array<Array<Double>>>(activityArray, valueArray)
    }

    private fun reshapeDataForTensorFlow(gyroSensorData: Array<Array<Double>>,
                                         accelSensorData: Array<Array<Double>>): FloatArray {
        val N_samples = gyroSensorData[0].size
        val out = FloatArray(size=6*N_samples, init = {0.0F})

        // Load Gyroscope Data
        for (k in 0 until 3) {
            for (l in 0 until N_samples) {
                val idx = k * N_samples + l
                out[idx] = gyroSensorData[k][l].toFloat()
            }
        }
        // Load Accelerometer Data
        for (k in 0 until 3) {
            for (l in 0 until N_samples) {
                val idx = (k+3) * N_samples + l
                out[idx] = accelSensorData[k][l].toFloat()
            }
        }

        return out
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

        return Pair(pretrainedModelOutArray, transferLearningModelOutArray)
    }

    private fun getActivityIdxWithHighestProbaility(probabilities: Array<Double>): Int {
        val N_values = probabilities.size
        val indices = (0 until N_values).toList().toIntArray()
        val sortedIndices = indices.sortedByDescending { idx -> probabilities[idx] }
        return sortedIndices[0]
    }

    private fun doResampling(gyroSensorData: Array<Array<Double>>,
                             accelSensorData: Array<Array<Double>>,
                             fs:Double=100.0,
                             Ns:Int = -1): Array<Array<Array<Double>>> {

        val gyroSensorDataResampled = SignalProcessingUtilities.resampleTimeSeries(gyroSensorData, fs=fs, N_samples_out = Ns)
        val accelSensorDataResampled = SignalProcessingUtilities.resampleTimeSeries(accelSensorData, fs=fs, N_samples_out = Ns)

        return arrayOf(gyroSensorDataResampled, accelSensorDataResampled)
    }

    private fun doNormalization(gyroSensorData: Array<Array<Double>>,
                                accelSensorData: Array<Array<Double>>): Array<Array<Array<Double>>> {
        val gyroSensorXMax: Double = gyroSensorData[0].maxOrNull() ?: 0.0
        val gyroSensorYMax: Double = gyroSensorData[1].maxOrNull() ?: 0.0
        val gyroSensorZMax: Double = gyroSensorData[2].maxOrNull() ?: 0.0

        val accelSensorXMax: Double = accelSensorData[0].maxOrNull() ?: 0.0
        val accelSensorYMax: Double = accelSensorData[1].maxOrNull() ?: 0.0
        val accelSensorZMax: Double = accelSensorData[2].maxOrNull() ?: 0.0

        val gyroX_normalized = gyroSensorData[0].map { elem ->  elem / gyroSensorXMax }.toTypedArray()
        val gyroY_normalized = gyroSensorData[1].map { elem ->  elem / gyroSensorYMax }.toTypedArray()
        val gyroZ_normalized = gyroSensorData[2].map { elem ->  elem / gyroSensorZMax }.toTypedArray()

        val accelX_normalized = accelSensorData[0].map { elem ->  elem / accelSensorXMax }.toTypedArray()
        val accelY_normalized = accelSensorData[1].map { elem ->  elem / accelSensorYMax }.toTypedArray()
        val accelZ_normalized = accelSensorData[2].map { elem ->  elem / accelSensorZMax }.toTypedArray()

        return arrayOf(
            arrayOf(gyroX_normalized, gyroY_normalized, gyroZ_normalized),
            arrayOf(accelX_normalized, accelY_normalized, accelZ_normalized))
    }

    private fun clipData(
        gyroSensorData: Array<Array<Double>>,
        accelSensorData: Array<Array<Double>>,
        T_lim:Double=10.0): Array<Array<Array<Double>>> {
        val clippedGyroData = SignalProcessingUtilities.clipDataSeries(gyroSensorData, T_clip=T_lim);
        val clippedAccelData = SignalProcessingUtilities.clipDataSeries(accelSensorData, T_clip=T_lim);

        return arrayOf(
            clippedGyroData,
            clippedAccelData)
    }

    private fun loadTestSet() :MutableList<TestSetSample>{
        val testSetFiles = activityThread.assets.list("test_set")
        val NTestsetFiles:Int = testSetFiles?.size ?: 0

        val testSetData = mutableListOf<TestSetSample>()
        for (k in 0 until NTestsetFiles) {
            val sample_filename:String = testSetFiles?.get(k)?: ""
            val sample_stream = activityThread.assets.open("test_set/%s".format(sample_filename))
            testSetData.add(loadTestsetSample(sample_stream, sample_filename))
        }
        return testSetData
    }

    private fun loadTestsetSample(stream: InputStream, filename:String, delimiter:String=";"): TestSetSample {

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

class TFLDataCapturingThread
constructor(val activityThread: TransferLearning): Thread() {
    override fun run() {

        waitUntilDataCaptureEnabled()
        flushSensorDataBuffers()
        waitUntilDataCaptureDisabled()

        // When locked, toggling the UI-button has not effect, except on its state-variable in the
        // activity-thread
        activityThread.transferLearningDataCapturingLock.lock()
        try {
            // Deepcopy the current values from the sensor-data buffers in a thread-save way
            val (gyroSensorData, accelSensorData) = copyBuffers()
            doProcessing(gyroSensorData, accelSensorData)
        } finally {
            activityThread.transferLearningDataCapturingLock.unlock()
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

    fun copyBuffers(): Pair<Array<Array<Double>>, Array<Array<Double>>> {
        // Copy the sensor-data arrays
        val gyroSensorData: Array<Array<Double>>;
        val accelSensorData: Array<Array<Double>>;

        // Interrupt writing new sensor-data to the buffers
        activityThread.writeDataLock.lock()
        try {
            gyroSensorData = SignalProcessingUtilities.deepcopySensorDataBuffer(activityThread.gyroSensorArray);
            accelSensorData = SignalProcessingUtilities.deepcopySensorDataBuffer(activityThread.accelSensorArray);
        } finally {
            activityThread.writeDataLock.unlock()
        }

        return Pair(gyroSensorData, accelSensorData)
    }

    fun doProcessing(gyroSensorData: Array<Array<Double>>, accelSensorData: Array<Array<Double>>) {

        val (gyroDataClipped, accelDataClipped) = clipData(gyroSensorData, accelSensorData)
        val (gyroDataResampled, accelDataResampled) = resampleData(gyroDataClipped, accelDataClipped)
        val timeframes = splitIntoTimeframes(gyroDataResampled, accelDataResampled)

        val k = 5
    }

    fun clipData(gyroSensorData: Array<Array<Double>>, accelSensorData: Array<Array<Double>>): Pair<Array<Array<Double>>, Array<Array<Double>>> {
        val gyroDataClipped = SignalProcessingUtilities.clipDataSeriesBackoff(gyroSensorData, T_backoff = 1.0)
        val accelDataClipped = SignalProcessingUtilities.clipDataSeriesBackoff(accelSensorData, T_backoff = 1.0)

        return Pair(gyroDataClipped, accelDataClipped)
    }

    fun resampleData(gyroSensorData: Array<Array<Double>>, accelSensorData: Array<Array<Double>>): Pair<Array<Array<Double>>, Array<Array<Double>>> {
        val gyroDataResampled = SignalProcessingUtilities.resampleTimeSeries(gyroSensorData, fs=50.0)
        val accelDataResampled = SignalProcessingUtilities.resampleTimeSeries(accelSensorData, fs=50.0)

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
}

class TFLClassificationTimerTask
constructor(val classificationObject: TFLClassificationThread, val activityThread: TransferLearning): TimerTask() {

    override fun run() {

        // The activitie's main-thread acquires the lock below to lock the classification process.
        if (activityThread.classificationLock.tryLock()) {
            // Trigger the classification
            classificationObject.synchronizationLock.lock()
            try {
                //classificationObject.synchronizationCondition.signal()
            } finally {
                classificationObject.synchronizationLock.unlock()
                activityThread.classificationLock.unlock()
            }
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
        fun clipDataSeries(dataArray: Array<Array<Double>>, T_clip:Double=10.0): Array<Array<Double>> {
            val T_high = dataArray[0][0];
            val N_samples = dataArray[0].size

            val T_low = T_high - T_clip

            // The lowest value >= T_high is searched. Remember that the array contains
            // the samples in descending order. (--> The firs row contains the latest sample.)
            var idx_T_end: Int = -1
            for (k in 0 until N_samples) {
                // Reached a negative time-value --> Too less data in buffer.
                if (dataArray[0][k] < 0) {
                    idx_T_end = k - 1
                    break;
                }
                if (dataArray[0][k] < T_high) {
                    idx_T_end = k - 1;
                    break;
                }
            }


            // The first value <= T_low is searched. Remember that the array contains
            // the samples in descending order. (--> The firs row contains the latest sample.)
            var idx_T_start: Int = -1
            for (k in idx_T_end until N_samples) {
                // Reached a negative time-value --> Too less data in buffer.
                if (dataArray[0][k] < 0) {
                    idx_T_start = k - 1
                    break;
                }
                if (dataArray[0][k] <= T_low) {
                    idx_T_start = k
                    break;
                }
            }


            // Reverse the row-order to have the lowest time-value first.
            return arrayOf(
                dataArray[0].slice(idx_T_start downTo idx_T_end).toTypedArray(),
                dataArray[1].slice(idx_T_start downTo idx_T_end).toTypedArray(),
                dataArray[2].slice(idx_T_start downTo idx_T_end).toTypedArray(),
                dataArray[3].slice(idx_T_start downTo idx_T_end).toTypedArray(),
            )
        }

        fun clipDataSeriesBackoff(dataArray: Array<Array<Double>>, T_backoff:Double=0.0): Array<Array<Double>> {
            val T_max = dataArray[0][0]

            // If the array is completely filled, T_lim contains a value >= 0. If not, its value is
            // <=0. In the latter case, the lowest time-value in the array is searched by iterating
            // over the timesteps.
            var T_min: Double = dataArray[0].last()
            // If T_min < 0 (the last row in the buffer is empty), the variable below holds the
            // index of the lowest time-value in the buffer.
            // If T_min >= 0, it simply holds the index of the last element in the buffer.
            var T_min_idx: Int = -1

            val T_high = T_max - T_backoff
            var T_high_idx:Int = -1

            var T_low: Double = Double.NaN
            var T_low_idx:Int = -1

            val N_samples = dataArray[0].size


            // Search for the first time-value < T_high.
            for (k in 0 until N_samples) {
                // A negative time-value implies that the current and all upcoming rows are unfilled.
                // --> The array contains too less data. Return empty arrays instead.
                if (dataArray[0][k] < 0) {
                    return arrayOf(
                        Array<Double>(0, init = {Double.NaN}),
                        Array<Double>(0, init = {Double.NaN}),
                        Array<Double>(0, init = {Double.NaN}),
                        Array<Double>(0, init = {Double.NaN}))
                }
                // Store the index of the first value <= T_high
                if (dataArray[0][k] <= T_high) {
                    T_high_idx = k
                    break;
                }
            }

            // Search for the lowest time-value in the buffer if it is not completely filled
            if (T_min < 0) {
                for (k in T_high_idx until N_samples) {
                    if (dataArray[0][k] < 0) {
                        T_min_idx = k - 1;
                        T_min = dataArray[0][T_min_idx]
                        break;
                    }
                }
            } else {
                T_min_idx = dataArray[0].lastIndex
            }

            // If the time-interval contained in the array is large enough, find the indices oof
            // the corresponding upper and lower boundaries. Otherwise, return an empty array.
            if (T_max - T_min < 2 * T_backoff) {
                return arrayOf(
                    Array<Double>(0, init = { Double.NaN}),
                    Array<Double>(0, init = { Double.NaN}),
                    Array<Double>(0, init = { Double.NaN}),
                    Array<Double>(0, init = { Double.NaN}))
            }

            T_low = T_min + T_backoff
            // Iterate from the minimum time-value in direction of increasing values to find the
            // first time-value above the T_low boundary.
            for (k in T_min_idx downTo T_high_idx) {
                if (dataArray[0][k] >= T_low) {
                    T_low_idx = k
                    break;
                }
            }

            // Reverse the row-order to have the lowest time-value first.
            return arrayOf(
                dataArray[0].slice(T_low_idx downTo T_high_idx).toTypedArray(),
                dataArray[1].slice(T_low_idx downTo T_high_idx).toTypedArray(),
                dataArray[2].slice(T_low_idx downTo T_high_idx).toTypedArray(),
                dataArray[3].slice(T_low_idx downTo T_high_idx).toTypedArray()
            )
        }

        fun resampleTimeSeries(dataArray: Array<Array<Double>>,
                               fs:Double=100.0,
                               N_samples_out:Int = -1): Array<Array<Double>> {
            val t_old = dataArray[0]
            val t0 = t_old[0]
            val t_new = t_old.map { tk -> tk - t0 };

            var __N_samples_out: Int = -1
            if (N_samples_out > 0) {
                __N_samples_out = N_samples_out
            } else {
                __N_samples_out = (t_new[t_new.size - 1]* fs).toInt();
            }

            val x1 = Array<Double>(size=__N_samples_out, init={0.0});
            val x2 = Array<Double>(size=__N_samples_out, init={0.0});
            val x3 = Array<Double>(size=__N_samples_out, init={0.0});

            var i: Int = 0;

            for (k: Int in 0 until __N_samples_out - 1) {

                val tk = k / fs;

                if (tk > t_new[i + 1]) {
                    i++;
                }

                val ti = t_new[i];
                // Interpolate
                val tip1 = t_new[i + 1];

                val x1_i = dataArray[1][i];
                val x1_ip1 = dataArray[1][i + 1];

                val x2_i = dataArray[2][i];
                val x2_ip1 = dataArray[2][i + 1];

                val x3_i = dataArray[3][i];
                val x3_ip1 = dataArray[3][i + 1];


                /* >>===== Linear interpolation for time-point tk between time-points ti and tip1 */

                // Compute this value only once, as it is required for all three data-channels.
                val delta_t = (tk - ti) / (tip1 - ti);

                x1[k] = (x1_ip1 - x1_i) * delta_t + x1_i;
                x2[k] = (x2_ip1 - x2_i) * delta_t + x2_i;
                x3[k] = (x3_ip1 - x3_i) * delta_t + x3_i;
                /* <<===== */
            }

            return arrayOf(x1, x2, x3);
        }

        fun deepcopySensorDataBuffer(buffer: Array<Array<Double>>): Array<Array<Double>> {
            val arraySize = buffer[0].size

            val arrayOut: Array<Array<Double>> = arrayOf(
                Array<Double>(size = arraySize, init = {idx -> buffer[0][idx]}),
                Array<Double>(size = arraySize, init = {idx -> buffer[1][idx]}),
                Array<Double>(size = arraySize, init = {idx -> buffer[2][idx]}),
                Array<Double>(size = arraySize, init = {idx -> buffer[3][idx]})
            )

            return arrayOut
        }
    }
}
