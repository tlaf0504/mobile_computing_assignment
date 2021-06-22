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
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import java.io.*
import java.lang.NumberFormatException
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
    private lateinit var bDebug: Button;

    /* ========== Sensor-data arrays ========== */
    val arraySize:Int = 10_000 // Array-sizes determined empirically
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

    //Handler triggered after model-inference
    lateinit var probabilityUpdateHandler: Handler;

    val activity_labels: Array<String> = arrayOf(
        "Biceps-Curls",
        "Triceps-Curls",
        "Russian-Twist",
        "Crunches"
    )

    // A lock acquired while writing data to the sensor-data arrays.
    // Acquire to lock data capturing.
    val writeDataLock = ReentrantLock()

    // Acquire to lock classification.
    val classificationLock = ReentrantLock()


    private lateinit var tricepsActivityTextView: TextView;
    private lateinit var bicepsActivityTextView: TextView;
    private lateinit var crunchesActivityTextView: TextView;
    private lateinit var russianTwistActivityTextView: TextView;
    private lateinit var classifiedActivityTextView: TextView;


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_recognition)

        probabilityUpdateHandler =
            Handler(mainLooper, Handler.Callback { msg -> probabilityUpdateHandlerCallback(msg) })

        bReturnToMainButton = findViewById(R.id.button_activity_return);
        bReturnToMainButton.setOnClickListener(this)

        tricepsActivityTextView = findViewById(R.id.activity_knn_1_prob)
        bicepsActivityTextView = findViewById(R.id.activity_knn_2_prob)
        crunchesActivityTextView = findViewById(R.id.activity_knn_3_prob)
        russianTwistActivityTextView = findViewById(R.id.activity_knn_4_prob)
        classifiedActivityTextView = findViewById(R.id.classified_activity_knn)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager;
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        
        classificationThread = ClassificationThread(this);
        classificationTimer = Timer();
        classificationTimerTask = ClassificationTimerTask(classificationThread);

        classificationThread.start()

        initUI()
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

    private fun initUI() {
        val values: Array<Double> = Array<Double>(size=activity_labels.size, init = {0.0})
        setActivityProbabilities(0, values)

    }

    fun probabilityUpdateHandlerCallback(msg: Message): Boolean {
        val (idx, values) = msg.obj as Pair<Int, Array<Double>>;
        setActivityProbabilities(idx, values)
        return true
    }

    fun setActivityProbabilities(mostProbableActivityIdx: Int, values:Array<Double>) {
        tricepsActivityTextView.text = resources.getString(R.string.activity_knn_1_prob, values[0])
        russianTwistActivityTextView.text = resources.getString(R.string.activity_knn_2_prob, values[1])
        bicepsActivityTextView.text = resources.getString(R.string.activity_knn_3_prob, values[2])
        crunchesActivityTextView.text = resources.getString(R.string.activity_knn_4_prob, values[3])

        classifiedActivityTextView.text = resources.getString(R.string.classified_activity_knn,
                activity_labels[mostProbableActivityIdx])
    }

    override fun onClick(v: View?) {
        if (v?.id == this.bReturnToMainButton.id) {
            val intent = Intent(this, MainActivity::class.java);
            startActivity(intent);
        }
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


class ClassificationThread
    constructor(val activityThread: ActivityRecognition) : Thread() {
    // Set to true for enabling the test-mode.
    // Thereby, provided samples used instead of the on-device sensor-data.
    val TEST:Boolean = true
    lateinit var testset: MutableList<TestSetSample>;
    var testsetCounter: Int = 0;

    val synchronizationLock = ReentrantLock()
    val synchronizationCondition = synchronizationLock.newCondition()


    // Set this to true to terminate the thread.
    var terminateThread: Boolean = false;
    private lateinit var neighbors: Array<Array<Double>>;
    private lateinit var neighbor_classes: Array<Int>;

    override fun run() {

        // Load the reference-data for kNN
        val reader = ReferenceDataCsvReader()
        val reference_data: Pair<Array<Array<Double>>, Array<Int>> =
            reader.read(activityThread.assets.open("kNN_reference_data.csv"))
        neighbors = reference_data.first;
        neighbor_classes = reference_data.second;

        if (TEST) {
            testset =
                TFLClassificationThread.loadSampleSet("knn_test_set", activityThread.assets)
            testsetCounter = 0;
        }

        while(!terminateThread) {


            // Wait until the timer-task triggers the classification
            synchronizationLock.lock()
            synchronizationCondition.await()

            // Arrays filled with the sensor-data, either with runtime-data for testset-data
            var gyroData: SensorData;
            var accelData: SensorData;

            activityThread.writeDataLock.lock()
            try {
                /*
                val stream_gyro = activityThread.assets.open("raw_sensor_data/curls/Curls_gyro_sensor_data_1624264514006_original_clipped.csv")
                gyroData = activityThread.loadRawSensorDataFromCSV(stream_gyro)

                val stream_accel = activityThread.assets.open("raw_sensor_data/curls/Curls_accel_sensor_data_1624264514006_original_clipped.csv")
                accelData = activityThread.loadRawSensorDataFromCSV(stream_accel)
                */

                gyroData = SignalProcessingUtilities.deepcopySensorDataBuffer(activityThread.gyroTimeArray, activityThread.gyroSensorArray)
                accelData = SignalProcessingUtilities.deepcopySensorDataBuffer(activityThread.accelTimeArray, activityThread.accelSensorArray)
            } finally {
                activityThread.writeDataLock.unlock()
            }

            // Do the computation with kNN classifier

            val (activity, probabilities) = doComputationkNN(gyroData, accelData)
            // <activity> < 0 in case of too less sensor-data (e.g. right after starting the app)
            if (activity >= 0) {
                val msg = Message()
                msg.obj = Pair(activity, probabilities)
                activityThread.probabilityUpdateHandler.sendMessage(msg)

                if (TEST) {
                    print("Cnt %d: Expected: %2d\tPredicted: %2d\n".format(testsetCounter, testset[testsetCounter].label, activity))
                    testsetCounter = (testsetCounter + 1) % testset.size
                }
            }

        }
    }

    private fun doComputationkNN(gyroData: SensorData,
                                 accelData: SensorData): Pair<Int, Array<Double>> {

        // Clip the time-series to an interval of 4 seconds. (Times are given in nanoseconds.)
        val T_clip: Long = 4_000_000_000
        val (gyroDataClipped, accelDataClipped) =
            clipData(gyroData, accelData, T_clip = T_clip);

        // Too less data available
        if (gyroDataClipped.first.isEmpty() || accelDataClipped.first.isEmpty()) {
            return Pair(-1, Array<Double>(size=4, init={0.0}))
        }


        val T_start_gyro: Long = gyroDataClipped.first[0]
        val T_end_gyro: Long = gyroDataClipped.first.last()
        val T_start_accel: Long = accelDataClipped.first[0]
        val T_end_accel: Long = accelDataClipped.first.last()

        // Captured Data-Frame to short
        if (T_end_gyro - T_start_gyro < T_clip ||
            T_end_accel - T_start_accel < T_clip) {
            // When too less data is available, return dummy values.
            return Pair(-1, Array<Double>(size=4, init={0.0}))
        }

        val resampledSensorData: Array<Array<Array<Double>>>;
        if (TEST) {
            resampledSensorData = arrayOf(
                testset[testsetCounter].gyroSensorData,
                testset[testsetCounter].accelSensorData)
        } else {
            resampledSensorData = doResampling(
                gyroDataClipped,
                accelDataClipped,
                fs = 50.0,
                Ns = 200
            )
        }

        val sensorDataWithoutOffset = removeOffsets(resampledSensorData[0], resampledSensorData[1])
        val energies = calcEnergies(sensorDataWithoutOffset[0], sensorDataWithoutOffset[1])
        return doClassification(energies[0], energies[1])
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

        val x1_squared: Array<Double> = dataArray[0].map { x -> x * x }.toTypedArray();
        val x2_squared: Array<Double> = dataArray[1].map { x -> x * x }.toTypedArray();
        val x3_squared: Array<Double> = dataArray[2].map { x -> x * x }.toTypedArray();

        return arrayOf(
                x1_squared.sum() / N_samples,
                x2_squared.sum() / N_samples,
                x3_squared.sum() / N_samples
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

        return kMajorityVoting(classes_of_nearest_neighbors, 10)
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

    fun read(stream: InputStream) : Pair<Array<Array<Double>>, Array<Int>> {

        val referenceData = mutableListOf<Array<Double>>();
        val referenceClasses = mutableListOf<Int>();

        val reader = stream.bufferedReader()
        var line: String? = reader.readLine();
        while (line != null) {
            val tokens = line.split(";")
            val tmp = arrayOf<Double>(
                    tokens[0].toDouble(), // Gyro x
                    tokens[1].toDouble(), // Gyro y
                    tokens[2].toDouble(), // Gyro z
                    tokens[3].toDouble(), // Accelerometer x
                    tokens[4].toDouble(), // Accelerometer y
                    tokens[5].toDouble(), // Accelerometer z
            )
            referenceData.add(element = tmp);
            referenceClasses.add(element = tokens[6].toInt());
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