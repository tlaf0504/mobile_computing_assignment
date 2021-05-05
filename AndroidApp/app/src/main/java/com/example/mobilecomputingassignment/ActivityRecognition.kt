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
import android.util.Range
import java.util.*
import java.lang.Thread
import android.view.View
import android.widget.Button
import android.widget.TextView
import com.example.mobilecomputingassignment.ml.ConvertedModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.lang.NumberFormatException
import java.nio.ByteBuffer
import java.nio.file.Path
import java.nio.file.Paths
import java.util.concurrent.locks.*
import java.util.regex.Pattern
import kotlin.math.pow




class ActivityRecognition : AppCompatActivity(), SensorEventListener, View.OnClickListener {

    private lateinit var sensorManager: SensorManager;
    private lateinit var gyroSensor: Sensor;
    private lateinit var accelSensor: Sensor;
    private lateinit var classificationThread: ClassificationThread;
    private lateinit var classificationTimer: Timer;
    private lateinit var classificationTimerTask: ClassificationTimerTask;
    private lateinit var bReturnToMainButton: Button;

    val class_labels: Array<String> = arrayOf(
            "Triceps-Curls",
            "Russian-Twist",
            "Biceps-Curls",
            "Crunches"
    )

    private lateinit var tricepsActivityTextView: TextView;
    private lateinit var bicepsActivityTextView: TextView;
    private lateinit var crunchesActivityTextView: TextView;
    private lateinit var russianTwistActivityTextView: TextView;


    /* ========== Sensor-data arrays ========== */

    // Assuming sensor events every 150ms, an sensor-array of length 70 should be long engough
    // to hold the desired timeframe of 10s. To have enough bachup, an array-length of 100 is
    // chosen.
    val arraySize:Int = 100
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

    val writeDataLock = ReentrantLock()
    // The single buffer-operations like switchBufferSet() or cleanInactiveBufferSet()
    // are used prevent race-conditions between them.
    private val bufferOperationLock = ReentrantLock()
    private var gyroDataCounter: Int = 0;
    private var accelDataCounter: Int = 0;

    lateinit var referenceDataFile : String;
    var testsetDirectory: String = "";

    lateinit var probabilityUpdateHandler:  Handler;




    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_recognition)

        tricepsActivityTextView = findViewById(R.id.triceps_probability)
        bicepsActivityTextView = findViewById(R.id.biceps_probability)
        crunchesActivityTextView = findViewById(R.id.crunches_probability)
        russianTwistActivityTextView = findViewById(R.id.russian_twist_probability)

        probabilityUpdateHandler = Handler(mainLooper, Handler.Callback { msg -> probabilityUpdateHandlerCallback(msg) })


        referenceDataFile = "${filesDir}/reference/reference_data.csv";

        // Comment the line below for regular operation. If uncommented,
        // the application loads a defined test-set and starts the classification
        // on the test-data.
        //testsetDirectory = "${filesDir}/test_set"

        bReturnToMainButton = findViewById(R.id.button_activity_return);
        bReturnToMainButton.setOnClickListener(this)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager;
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        
        classificationThread = ClassificationThread(this);
        classificationTimer = Timer();
        classificationTimerTask = ClassificationTimerTask(classificationThread);

        classificationThread.start()
    }

    override fun onStart() {
        super.onStart()

        // Register the event-listeners for the Gyroscope and the Accelerometer.
        sensorManager.registerListener(this, gyroSensor, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, accelSensor, SensorManager.SENSOR_DELAY_NORMAL);

        // Schedule the classification task at a fixed rate of 2 seconds, but wait for 1 second
        // to start to ensure the UI is build up.
        classificationTimer.scheduleAtFixedRate(classificationTimerTask, 1000, 2000);
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

        for (k in N_entries - 1 downTo 1) {
            for (l in 0 until N_columns) {
                buffer[l][k] = buffer[l][k - 1]
            }
        }

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

    fun probabilityUpdateHandlerCallback(msg: Message):Boolean {
        setActivityProbabilities(msg.obj as Array<Double>)
        return true
    }

    fun setActivityProbabilities(values:Array<Double>) {
        tricepsActivityTextView.text = resources.getString(R.string.triceps_percentage, values[0])
        russianTwistActivityTextView.text = resources.getString(R.string.russian_twist_percentage, values[1])
        bicepsActivityTextView.text = resources.getString(R.string.biceps_percentage, values[2])
        crunchesActivityTextView.text = resources.getString(R.string.crunches_percentage, values[3])
    }

    override fun onClick(v: View?) {
        if (v?.id == this.bReturnToMainButton.id) {
            val intent = Intent(this, MainActivity::class.java);
            startActivity(intent);
        }
    }
}





class ClassificationThread
    constructor(val activityThread: ActivityRecognition) : Thread() {

    val synchronizationLock = ReentrantLock()
    val synchronizationCondition = synchronizationLock.newCondition()

    private lateinit var gyroTestData: List<Array<Array<Double>>>;
    private lateinit var accelTestData: List<Array<Array<Double>>>;
    private lateinit var testClasses: List<Int>;
    private var useTestData: Boolean = false;
    private var testsetSize: Int = -1;
    private var currentTestIndex: Int = -1;



    // Set this to true to terminate the thread.
    var terminateThread: Boolean = false;
    private val sampling_frequency: Double = 100.0;
    private lateinit var neighbors: Array<Array<Double>>;
    private lateinit var neighbor_classes: Array<Int>;


    private lateinit var model: ConvertedModel;
    private lateinit var inputFeature0: TensorBuffer;
    private val activity_labels: Array<String> = arrayOf(
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

    override fun run() {

        model = ConvertedModel.newInstance(activityThread);
        inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1200), DataType.FLOAT32);

        // Load the reference-data for kNN
        val reader = ReferenceDataCsvReader()
        val ret_vals:Pair<Array<Array<Double>>, Array<Int>> = reader.read(activityThread.referenceDataFile)
        neighbors = ret_vals.first;
        neighbor_classes = ret_vals.second;

        // Is the testsetDirectory-member in the Activity is not empty, iteratively apply the
        // pre-captured test-signals to the classification-algorithm instead of using the
        // sensor-data buffers.
        if (activityThread.testsetDirectory.isNotEmpty()) {
            val testSetLoader : TestSetLoader = TestSetLoader()
            val (tmp1, tmp2, tmp3) =
                    testSetLoader.load(activityThread.testsetDirectory);

            gyroTestData = tmp1;
            accelTestData = tmp2;
            testClasses = tmp3;

            useTestData = true;
            testsetSize = gyroTestData.size;
            currentTestIndex = 0;
        } else {
            useTestData = false;
        }


        while(!terminateThread) {


            // Wait until the timer-task triggers the classification
            synchronizationLock.lock()
            synchronizationCondition.await()

            // Arrays filled with the sensor-data, either with runtime-data for testset-data
            var gyroSensorData: Array<Array<Double>>;
            var accelSensorData: Array<Array<Double>>;

            // Only used when applying the testset-data to the classification algorithms.
            // Contains the expected class of the test-sample.
            var expected_activity: Int = -1;

            // Use test-data as classification-input
            if (useTestData) {

                gyroSensorData = arrayOf(
                        gyroTestData[currentTestIndex][0].reversedArray(),
                        gyroTestData[currentTestIndex][1].reversedArray(),
                        gyroTestData[currentTestIndex][2].reversedArray(),
                        gyroTestData[currentTestIndex][3].reversedArray()
                )

                accelSensorData = arrayOf(
                        accelTestData[currentTestIndex][0].reversedArray(),
                        accelTestData[currentTestIndex][1].reversedArray(),
                        accelTestData[currentTestIndex][2].reversedArray(),
                        accelTestData[currentTestIndex][3].reversedArray()
                )

                expected_activity = testClasses[currentTestIndex];



            // Run the classification with the actual sensor-data
            } else {
                activityThread.writeDataLock.lock()
                try {
                     gyroSensorData = deepcopySensorDataBuffer(activityThread.gyroSensorArray);
                     accelSensorData = deepcopySensorDataBuffer(activityThread.accelSensorArray);
                } finally {
                    activityThread.writeDataLock.unlock()
                }
            }

            // Do the computation with kNN classifier
            /*val (activity, probabilities) = doComputationkNN(gyroSensorData, accelSensorData)
            // <activity> < 0 in case of too less sensor-data (e.g. right after starting the app)
            if (activity >= 0) {
                val msg = Message()
                msg.obj = probabilities
                activityThread.probabilityUpdateHandler.sendMessage(msg)
            }*/

            val (activity, probabilities) = doComputationTFLite(gyroSensorData, accelSensorData)



            // When applying the testset-data, write the result to the system-console.
            if (useTestData) {
                println("Idx: ${currentTestIndex}\tExpected: ${expected_activity}\tCalculated: ${activity}")
                currentTestIndex = (currentTestIndex + 1) % testsetSize;
            }
        }
    }

    private fun doComputationTFLite(gyroSensorData: Array<Array<Double>>,
                                    accelSensorData: Array<Array<Double>>): Pair<Int,Array<Double>> {

        //Extract the available data from the buffer it is not filled completely
        val clippedData = clipData(gyroSensorData, accelSensorData, T_lim = 4.0);

        // Too less data available
        if (clippedData[0][0].last() - clippedData[0][0].first() < 4.0) {
            return Pair(-1, arrayOf(0.0, 0.0, 0.0, 0.0))
        }

        val resampledSensorData = doResampling(clippedData[0],
            clippedData[0],
            fs=50.0,
            Ns=200)


        var tmp: FloatArray = FloatArray(size=1200, init={0.0F})

        // Load Gyroscope Data
        for (k in 0 until 2) {
            for (l in 0 until 199) {
                val idx = k * 200 + l
                tmp[idx] = resampledSensorData[0][k][l].toFloat()
            }
        }
        // Load Accelerometer Data
        for (k in 3 until 5) {
            for (l in 0 until 199) {
                val idx = k * 200 + l
                tmp[idx] = resampledSensorData[1][k-3][l].toFloat()
            }
        }

        // Execute the model inference
        inputFeature0.loadArray(tmp)
        val outputs = model.process(inputFeature0)

        // Convert output-data
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val outputArray = outputFeature0.floatArray

        val outputIndexArray = (0 until 11).toList().toIntArray()
        val sortedOutputIndexArray = outputIndexArray.sortedBy { idx -> outputArray[idx] }

        for (k in 0 until 3) {
            print("%18s:  %4.2f  |  ".format(
                activity_labels[sortedOutputIndexArray[k]],
                outputArray[sortedOutputIndexArray[k]] * 100))
        }
        print("\n")



        val arr = Array<Double>(size=10, init={0.0})
        return Pair<Int, Array<Double>>(-1,arr)
    }

    private fun doComputationkNN(gyroSensorData: Array<Array<Double>>,
                              accelSensorData: Array<Array<Double>>): Pair<Int,Array<Double>> {

        val clippedData = clipData(gyroSensorData, accelSensorData);
        val N_samples_gyro = clippedData[0][0].size
        val N_samples_accel = clippedData[1][0].size

        // Too less sampled to provice proper classification
        if (N_samples_gyro < 10 || N_samples_accel < 10) {
            return Pair(-1, arrayOf(0.0, 0.0, 0.0, 0.0))
        }
        val resampledSensorData = doResampling(clippedData[0], clippedData[1])
        val sensorDataWithoutOffset = removeOffsets(resampledSensorData[0], resampledSensorData[1])
        val energies = calcEnergies(sensorDataWithoutOffset[0], sensorDataWithoutOffset[1])
        return doClassification(energies[0], energies[1])
    }

    private fun doResampling(gyroSensorData: Array<Array<Double>>,
                             accelSensorData: Array<Array<Double>>,
                             fs:Double=100.0,
                             Ns:Int = -1): Array<Array<Array<Double>>> {

        val gyroSensorDataResampled = resampleTimeSeries(gyroSensorData, fs=fs, N_samples_out = Ns)
        val accelSensorDataResampled = resampleTimeSeries(accelSensorData, fs=fs, N_samples_out = Ns)

        return arrayOf(gyroSensorDataResampled, accelSensorDataResampled)
    }

    private fun clipData(
            gyroSensorData: Array<Array<Double>>,
            accelSensorData: Array<Array<Double>>,
            T_lim:Double=10.0): Array<Array<Array<Double>>> {
        val clippedGyroData = clipDataSeries(gyroSensorData, T_lim=T_lim);
        val clippedAccelData = clipDataSeries(accelSensorData, T_lim=T_lim);

        return arrayOf(
                clippedGyroData,
                clippedAccelData)
    }

    private fun clipDataSeries(dataArray: Array<Array<Double>>, T_lim:Double=10.0): Array<Array<Double>> {
        val tmax = dataArray[0][0];
        val N_samples = dataArray[0].size

        // If the last row in the array is filled (indicated by a timestamp >= 0),
        // AND the timeframe is shorter than the clipping-length, on can return the whole array.
        if (dataArray[0].last() > 0 &&
            dataArray[0][0] - dataArray[0].last() < T_lim) {

            // Reverse the row-order to have the lowest time-value first.
            return arrayOf(
                dataArray[0].slice(dataArray[0].lastIndex downTo 0).toTypedArray(),
                dataArray[1].slice(dataArray[0].lastIndex downTo 0).toTypedArray(),
                dataArray[2].slice(dataArray[0].lastIndex downTo 0).toTypedArray(),
                dataArray[3].slice(dataArray[0].lastIndex downTo 0).toTypedArray(),
            )
        }

        // Go through the time-values and find the first one exceeding the given time-limit.
        // If a "-1" is found beforehand, terminate the loop and return the array until the pre-last
        // value.
        var t0_idx = -1;
        for (k in 0 until N_samples) {

            // If a -1 is found, set the end-index to k-1
            if (dataArray[0][k] < 0) {
                t0_idx = k - 1;
                break;
            }

            // If the limit-time is exceeded, use set the current row to be the last one in the
            // returned array.
            if (tmax - dataArray[0][k] >= T_lim) {
                t0_idx = k;
                break;
            }
        }

        // Reverse the row-order to have the lowest time-value first.
        return arrayOf(
                dataArray[0].slice(t0_idx downTo 0).toTypedArray(),
                dataArray[1].slice(t0_idx downTo 0).toTypedArray(),
                dataArray[2].slice(t0_idx downTo 0).toTypedArray(),
                dataArray[3].slice(t0_idx downTo 0).toTypedArray(),
        )
    }

    private fun resampleTimeSeries(dataArray: Array<Array<Double>>,
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


    private fun removeOffsets(gyroSensorData: Array<Array<Double>>,
                              accelSensorData: Array<Array<Double>>): Array<Array<Array<Double>>> {

        val gyroDataWithoutOffset = removeOffset(gyroSensorData)
        val accelDataWithoutOffset = removeOffset(accelSensorData)

        return arrayOf(gyroDataWithoutOffset, accelDataWithoutOffset)
    }

    private fun removeOffset(dataArray: Array<Array<Double>>): Array<Array<Double>> {

        val offset1 = dataArray[0].average();
        val offset2 = dataArray[1].average();
        val offset3 = dataArray[2].average();

        return arrayOf(
                dataArray[0].map { x -> (x - offset1).toDouble() }.toTypedArray(),
                dataArray[1].map { x -> (x - offset2).toDouble() }.toTypedArray(),
                dataArray[2].map { x -> (x - offset3).toDouble() }.toTypedArray()
        )
    }

    private fun calcEnergies(gyroSensorData: Array<Array<Double>>,
                             accelSensorData: Array<Array<Double>>): Array<Array<Double>> {

        val gyroSensorEnergies: Array<Double> = calcEnergy(gyroSensorData);
        val accelSensorEnergies: Array<Double> = calcEnergy(accelSensorData);

        return arrayOf(gyroSensorEnergies, accelSensorEnergies)
    }

    private fun calcEnergy(dataArray: Array<Array<Double>>): Array<Double> {
        val N_samples = dataArray[0].size;

        val x1_squared: Array<Double> = dataArray[0].map { x -> x * x / N_samples }.toTypedArray();
        val x2_squared: Array<Double> = dataArray[1].map { x -> x * x / N_samples}.toTypedArray();
        val x3_squared: Array<Double> = dataArray[2].map { x -> x * x / N_samples}.toTypedArray();

        return arrayOf(
                x1_squared.sum(),
                x2_squared.sum(),
                x3_squared.sum()
        )
    }

    private fun doClassification(gyroEnergies: Array<Double>, accelEnergies: Array<Double>): Pair<Int,Array<Double>> {
        val energy_vector: Array<Double> = arrayOf(
                gyroEnergies[0],
                gyroEnergies[1],
                gyroEnergies[2],
                accelEnergies[0],
                accelEnergies[1],
                accelEnergies[2]
        );

        //print("Gyro-X: ${gyroEnergies[0]}, Gyro-Y: ${gyroEnergies[1]}, Gyro-Z: ${gyroEnergies[2]}, Accel-X: ${gyroEnergies[0]}, Accel-Y: ${gyroEnergies[1]}, Accel-Z: ${gyroEnergies[2]}\n")

        // Calculate euclidean distances to reference individuals.
        val euclideanDistances = Array<Double>(size=neighbors.size, init={0.0});
        for (k: Int in 0 until neighbors.size) {
            euclideanDistances[k] = euclideanDistance(energy_vector, neighbors[k]);
        }

        val neighbor_indices: IntArray = (0..neighbors.size - 1).toList().toIntArray();
        val sorted_neighbor_indices = neighbor_indices.sortedBy { idx -> euclideanDistances[idx] }
        val classes_of_nearest_neighbors = neighbor_classes.sliceArray(sorted_neighbor_indices)

        return kMajorityVoting(classes_of_nearest_neighbors, 3)
    }

    private fun kMajorityVoting(sortedClassesOfNearestNeighbors: Array<Int>, k:Int): Pair<Int,Array<Double>> {
        // Classes have IDs from 0 to maxClass
        val maxClass: Int = (sortedClassesOfNearestNeighbors.maxOrNull()?: 0) + 1

        val classCounters = Array<Int>(size = maxClass, init = {0})

        for (l in 0 until k) {
            classCounters[sortedClassesOfNearestNeighbors[k]]++;
        }

        var _class_max: Int = -1;
        var _class: Int = -1;
        for (l in 0 until maxClass) {
            if (classCounters[l] > _class_max) {
                _class_max = classCounters[l];
                _class = l;
            }
        }

        // Calculate the probabilities for the current individual to belong to one of the single
        // reference-classes
        val neighborhood_probabilities =
                Array<Double>(size = classCounters.size, init = {idx -> classCounters[idx].toDouble() / k})

        return Pair(_class, neighborhood_probabilities)
    }

    private fun euclideanDistance(v1: Array<Double>, v2: Array<Double>): Double {
        val diff = Array<Double>(size = v1.size, init={0.0});

        // Squared differences
        for (k: Int in 0 until v1.size) {
            diff[k] = (v1[k] - v2[k]).pow(2);
        }

        // Square-root of sum
        return diff.sum().pow(x = 0.5);
    }

    private fun deepcopySensorDataBuffer(buffer: Array<Array<Double>>): Array<Array<Double>> {

        var arrayOut: Array<Array<Double>> = arrayOf(
                Array<Double>(size = activityThread.arraySize, init = {idx -> buffer[0][idx]}),
                Array<Double>(size = activityThread.arraySize, init = {idx -> buffer[1][idx]}),
                Array<Double>(size = activityThread.arraySize, init = {idx -> buffer[2][idx]}),
                Array<Double>(size = activityThread.arraySize, init = {idx -> buffer[3][idx]})
        )

        return arrayOut
    }
}

class ClassificationTimerTask
    constructor(val classificationObject: ClassificationThread): TimerTask() {

    override fun run() {
        // Trigger the classification
        classificationObject.synchronizationLock.lock()
        try {
            classificationObject.synchronizationCondition.signal()
        } finally {
            classificationObject.synchronizationLock.unlock()
        }
    }
}

class ReferenceDataCsvReader {

    fun read(file:String) : Pair<Array<Array<Double>>, Array<Int>> {

        val referenceData = mutableListOf<Array<Double>>();
        val referenceClasses = mutableListOf<Int>();

        val file_ = File(file);
        val reader = file_.bufferedReader()
        var line: String? = reader.readLine();
        while (line != null) {
            val tokens = line.split(";")
            val tmp = arrayOf<Double>(
                    tokens[1].toDouble(), // Gyro x
                    tokens[2].toDouble(), // Gyro y
                    tokens[3].toDouble(), // Gyro z
                    tokens[4].toDouble(), // Accelerometer x
                    tokens[5].toDouble(), // Accelerometer y
                    tokens[6].toDouble(), // Accelerometer z
            )
            referenceData.add(element = tmp);
            referenceClasses.add(element = tokens[7].toInt());
            line = reader.readLine()
        }

        reader.close()
        return Pair(referenceData.toTypedArray(), referenceClasses.toTypedArray())
    }
}

class TestSetLoader {

    private val activityDirectories =
            arrayOf<String>("triceps", "russian_twist", "curls", "crunches")

    // The activity-classes corresponding to the upper subdirectories.
    private val actitityClasses = arrayOf<Int>(0,1,2,3)

    fun load(testsetDirectory: String): Triple<List<Array<Array<Double>>>, List<Array<Array<Double>>>, List<Int>> {
        /* The test-set consists of CSV-files for each activity, with two files per activity. (One
        for the gyro-sensor and one for the accelerometer.)

        For each activity, the sensor-data files are located each in a subdirectory.

        The properties <activityDirectories> specify the subdirectories and <activityClasses>
        state the class-id for each subdirectory.

         */

        val gyroData: MutableList<Array<Array<Double>>> = mutableListOf();
        val accelData: MutableList<Array<Array<Double>>> = mutableListOf();
        val activityClassPerFilePair: MutableList<Int> = mutableListOf();

        val gyroRegexPattern: Pattern = Pattern.compile(".*_gyro_.*\\.csv");

        for (l in 0 until activityDirectories.size) {
            val subdir_abspath: Path = Paths.get(testsetDirectory, activityDirectories[l])
            val subdir_object: File = subdir_abspath.toFile()

            // Search for all gyro-files in the current directory
            val gyro_files_in_subdir = subdir_object.list { dir, name ->  gyroRegexPattern.matcher(name).matches()}
            // For each gyro-file, search for the corresponding accelerometer-file
            val accel_files_in_subdir = gyro_files_in_subdir.map { s -> s.replace("gyro", "accel") }

            // Iterate over all gyro-/accelerometer-file pairs and load their data.
            for (k in 0 until gyro_files_in_subdir.size) {
                // Read gyro sensor-data
                gyroData.add(readCSVSensorDataFile(Paths.get(testsetDirectory, activityDirectories[l], gyro_files_in_subdir[k]).toString()))
                // Read accelerometer sensor-data
                accelData.add(readCSVSensorDataFile(Paths.get(testsetDirectory, activityDirectories[l],accel_files_in_subdir[k]).toString()))
                // assign activity class
                activityClassPerFilePair.add(actitityClasses[l])

            }
        }

        return Triple<List<Array<Array<Double>>>, List<Array<Array<Double>>>, List<Int>> (gyroData, accelData, activityClassPerFilePair)
    }


    private fun readCSVSensorDataFile(file:String): Array<Array<Double>> {

        val data: MutableList<Array<Double>> = mutableListOf();

        val reader: BufferedReader = File(file).bufferedReader()
        var line: String? = reader.readLine();
        while (line != null) {
            val tokens = line.split(";")
            try {
                val tmp = arrayOf<Double>(
                        tokens[0].toDouble(), // Time
                        tokens[1].toDouble(), // Data x
                        tokens[2].toDouble(), // Data y
                        tokens[3].toDouble()  // Data z
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
                Array<Double>(N_samples, init = {0.0})
        )

        // Convert from row-indices first to column-indices first, to match the required data-format
        for (k in 0 until N_samples) {
            dataArray[0][k] = data[k][0]
            dataArray[1][k] = data[k][1]
            dataArray[2][k] = data[k][2]
            dataArray[3][k] = data[k][3]
        }
        return dataArray
    }
}