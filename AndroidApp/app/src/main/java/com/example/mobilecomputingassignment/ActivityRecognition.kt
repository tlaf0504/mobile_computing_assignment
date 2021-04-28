package com.example.mobilecomputingassignment

import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.service.autofill.FieldClassification
import kotlinx.coroutines.*
import java.util.*
import java.lang.Thread
import android.util.Log
import android.view.View
import android.widget.Button
import java.io.*
import java.lang.NumberFormatException
import java.nio.file.Path
import java.nio.file.Paths
import java.util.concurrent.locks.*
import java.util.concurrent.*
import java.util.regex.Matcher
import java.util.regex.Pattern
import kotlin.math.max
import kotlin.math.pow

class ActivityRecognition : AppCompatActivity(), SensorEventListener, View.OnClickListener {

    private lateinit var sensorManager: SensorManager;
    private lateinit var gyroSensor: Sensor;
    private lateinit var accelSensor: Sensor;
    private lateinit var classificationThread: ClassificationThread;
    private lateinit var classificationTimer: Timer;
    private lateinit var classificationTimerTask: ClassificationTimerTask;
    private lateinit var bReturnToMainButton: Button;

    /* ========== Sensor-data arrays ========== */
    
    val arraySize:Int = 1000
    val gyroSensorArray1 = arrayOf(
            Array<Float>(size=arraySize, init={-1.0F}),
            Array<Float>(size=arraySize, init={0.0F}),
            Array<Float>(size=arraySize, init={0.0F}),
            Array<Float>(size=arraySize, init={0.0F})
    );

    val accelSensorArray1 = arrayOf(
            Array<Float>(size=arraySize, init={-1.0F}),
            Array<Float>(size=arraySize, init={0.0F}),
            Array<Float>(size=arraySize, init={0.0F}),
            Array<Float>(size=arraySize, init={0.0F})
    );

    val gyroSensorArray2 = arrayOf(
            Array<Float>(size=arraySize, init={-1.0F}),
            Array<Float>(size=arraySize, init={0.0F}),
            Array<Float>(size=arraySize, init={0.0F}),
            Array<Float>(size=arraySize, init={0.0F})
    );

    val accelSensorArray2 = arrayOf(
            Array<Float>(size=arraySize, init={-1.0F}),
            Array<Float>(size=arraySize, init={0.0F}),
            Array<Float>(size=arraySize, init={0.0F}),
            Array<Float>(size=arraySize, init={0.0F})
    );

    private val writeDataLock = ReentrantLock()
    // The single buffer-operations like switchBufferSet() or cleanInactiveBufferSet()
    // are used prevent race-conditions between them.
    private val bufferOperationLock = ReentrantLock()
    private var gyroDataCounter1: Int = 0;
    private var gyroDataCounter2: Int = 0;
    private var accelDataCounter1: Int = 0;
    private var accelDataCounter2: Int = 0;
    private var activeBuffer : Int = 0;

    lateinit var referenceDataFile : String;
    var testsetDirectory: String = "";


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_recognition)


        referenceDataFile = "${filesDir}/reference/reference_data.csv";

        // Comment the line below for regular operation. If uncommented,
        // the application loads a defined test-set and starts the classification
        // on the test-data.
        testsetDirectory = "${filesDir}/test_set"

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
            val gyroX : Float = event.values[0];
            val gyroY : Float = event.values[1];
            val gyroZ : Float = event.values[2];
            // Get the timestamp in seconds
            val time: Float = event.timestamp.toFloat() * 1e-9F;

            writeDataLock.lock()
            try {

                if (activeBuffer == 1 && gyroDataCounter1 < gyroSensorArray1[0].size) {
                    gyroSensorArray1[0][gyroDataCounter1] = time
                    gyroSensorArray1[1][gyroDataCounter1] = gyroX
                    gyroSensorArray1[2][gyroDataCounter1] = gyroY
                    gyroSensorArray1[3][gyroDataCounter1] = gyroZ

                    gyroDataCounter1++;

                } else if (gyroDataCounter2 < gyroSensorArray2[0].size) {
                    gyroSensorArray2[0][gyroDataCounter2] = time
                    gyroSensorArray2[1][gyroDataCounter2] = gyroX
                    gyroSensorArray2[2][gyroDataCounter2] = gyroY
                    gyroSensorArray2[3][gyroDataCounter2] = gyroZ

                    gyroDataCounter2++;
                }
            } finally {
                writeDataLock.unlock()
            }

        } else if (sensorType == Sensor.TYPE_ACCELEROMETER) {
            val accelX : Float = event.values[0];
            val accelY : Float = event.values[1];
            val accelZ : Float = event.values[2];
            // Get the timestamp in seconds
            val time: Float = event.timestamp.toFloat() * 1e-9F;

            writeDataLock.lock()
            try {

                if (activeBuffer == 1 && accelDataCounter1 < accelSensorArray1[0].size) {
                    accelSensorArray1[0][accelDataCounter1] = time
                    accelSensorArray1[1][accelDataCounter1] = accelX
                    accelSensorArray1[2][accelDataCounter1] = accelY
                    accelSensorArray1[3][accelDataCounter1] = accelZ

                    accelDataCounter1++;

                } else if (accelDataCounter2 < accelSensorArray2[0].size) {
                    accelSensorArray2[0][accelDataCounter2] = time
                    accelSensorArray2[1][accelDataCounter2] = accelX
                    accelSensorArray2[2][accelDataCounter2] = accelY
                    accelSensorArray2[3][accelDataCounter2] = accelZ

                    accelDataCounter2++;
                }
            } finally {
                writeDataLock.unlock()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Mandatory implementation by child-class of SensorEventListener()
    }

    fun switchBufferSet() {
        /* Attention: This function DOES NOT clean up the buffer or reset the index-counters. Use
        cleanUpInactiveBufferSet() for this purpose. */

        // Block any concurrent buffer-operations. (Capturing sensor-data is still enabled.)
        bufferOperationLock.lock();

        //Block sensor-data capturing
        writeDataLock.lock();

        // Now, switch to the inactive buffer set.
        try {
            if (activeBuffer == 0) {
                activeBuffer = 1;
            } else {
                activeBuffer = 0;
            }
        } finally {
            // Re-enable data capturing.
            writeDataLock.unlock();

            // Release lock for concurrent buffer operations.
            bufferOperationLock.unlock();
        }
    }

    fun getActiveBufferSet() : Array<Array<Array<Float>>> {
        writeDataLock.lock()
        bufferOperationLock.lock()

        // As Kotlin currently does not any easy to use deep-copy mechanism,
        // one has to use the rather complicated implementation below.
        val res : Array<Array<Array<Float>>> = arrayOf(
                arrayOf(Array(arraySize, {0.0F}),
                        Array(arraySize, {0.0F}),
                        Array(arraySize, {0.0F}),
                        Array(arraySize, {0.0F})),
                arrayOf(Array(arraySize, {0.0F}),
                        Array(arraySize, {0.0F}),
                        Array(arraySize, {0.0F}),
                        Array(arraySize, {0.0F}))
        )

        try {
            if (activeBuffer == 0) {
                arrayOf(gyroSensorArray1, accelSensorArray1).copyInto(res)
            } else {
                arrayOf(gyroSensorArray2, accelSensorArray2).copyInto(res)
            }
        } finally {
            bufferOperationLock.unlock()
            writeDataLock.unlock()
        }

        return res
    }

    fun getInactiveBufferSet() : List<Array<Array<Float>>> {

        // Block any concurrent buffer-operations.
        bufferOperationLock.lock()

        val res: List<Array<Array<Float>>>;
        try {
            if (activeBuffer == 0) {
                 res = listOf(gyroSensorArray2, accelSensorArray2)
            } else {
                res = listOf(gyroSensorArray1, accelSensorArray1)
            }
        } finally {
            bufferOperationLock.unlock()
        }

        return res
    }
    
    fun cleanInactiveBufferSet() {
        // Block any concurrent buffer-operations. (Data-capturing is still enabled.)
        bufferOperationLock.lock()

        try {
            // As bufferOperationLock is locked, one can safely query activeBuffer.
            if (activeBuffer == 1) {
                cleanBuffer(gyroSensorArray1);
                cleanBuffer(accelSensorArray1);
                gyroDataCounter1 = 0;
                accelDataCounter1 = 0;
            } else {
                cleanBuffer(gyroSensorArray2);
                cleanBuffer(accelSensorArray2);
                gyroDataCounter2 = 0;
                accelDataCounter2 = 0;
            }
        } finally {
            bufferOperationLock.lock()
        }
    }
    
    private fun cleanBuffer(buffer: Array<Array<Float>>) {
        buffer[0].fill(element=-1.0F)
        buffer[1].fill(element=0.0F)
        buffer[2].fill(element=0.0F)
        buffer[3].fill(element=0.0F)
        
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

    private lateinit var gyroTestData: List<Array<Array<Float>>>;
    private lateinit var accelTestData: List<Array<Array<Float>>>;
    private lateinit var testClasses: List<Int>;
    private var useTestData: Boolean = false;
    private var testsetSize: Int = -1;
    private var currentTestIndex: Int = -1;

    // Set this to true to terminate the thread.
    var terminateThread: Boolean = false;

    private val sampling_frequency: Float = 100.0F;

    // ToDo: Replace space-holder neighbors with real ones
    private lateinit var neighbors: Array<Array<Float>>;

    private lateinit var neighbor_classes: Array<Int>;

    override fun run() {

        val reader = ReferenceDataCsvReader()
        val ret_vals:Pair<Array<Array<Float>>, Array<Int>> = reader.read(activityThread.referenceDataFile)
        neighbors = ret_vals.first;
        neighbor_classes = ret_vals.second;

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

        }


        while(!terminateThread) {


            // Wait until the timer-task triggers the classification
            synchronizationLock.lock()
            synchronizationCondition.await()

            if (useTestData) {
                val activity = doComputation(gyroTestData[currentTestIndex],
                        accelTestData[currentTestIndex])
                val expected_activity = testClasses[currentTestIndex];

                println("Idx: ${currentTestIndex}\tExpected: ${expected_activity}\tCalculated: ${activity}")
                currentTestIndex = (currentTestIndex + 1) % testsetSize;

            } else {

                // Switch to the inactive buffer for storing the sensor-data.
                activityThread.switchBufferSet()

                // Get the new inactive buffer-set for processing.
                val sensorData = activityThread.getInactiveBufferSet()

                // Do the computation
                val activity = doComputation(sensorData[0], sensorData[1])

                // Clean-up the currently inactive buffer to be ready for filling after the next
                // timer-event.
                activityThread.cleanInactiveBufferSet()
            }
        }
    }

    private fun doComputation(gyroSensorData: Array<Array<Float>>,
                              accelSensorData: Array<Array<Float>>): Int {

        val clippedData = clipData(gyroSensorData, accelSensorData);
        val N_samples_gyro = clippedData[0][0].size
        val N_samples_accel = clippedData[1][0].size

        // Too less sampled to provice proper classification
        if (N_samples_gyro < 10 || N_samples_accel < 10) {
            return -1
        }
        val resampledSensorData = doResampling(clippedData[0], clippedData[1])
        val sensorDataWithoutOffset = removeOffsets(resampledSensorData[0], resampledSensorData[1])
        val energies = calcEnergies(sensorDataWithoutOffset[0], sensorDataWithoutOffset[1])
        val res = doClassification(energies[0], energies[1])
        return res
    }

    private fun doResampling(gyroSensorData: Array<Array<Float>>,
                             accelSensorData: Array<Array<Float>>): Array<Array<Array<Float>>> {

        val gyroSensorDataResampled = resampleTimeSeries(gyroSensorData)
        val accelSensorDataResampled = resampleTimeSeries(accelSensorData)

        return arrayOf(gyroSensorDataResampled, accelSensorDataResampled)
    }

    private fun clipData(
            gyroSensorData: Array<Array<Float>>,
            accelSensorData: Array<Array<Float>>): Array<Array<Array<Float>>> {
        val clippedGyroData = clipDataSeries(gyroSensorData);
        val clippedAccelData = clipDataSeries(accelSensorData);

        return arrayOf(
                clippedGyroData,
                clippedAccelData)
    }

    private fun clipDataSeries(dataArray: Array<Array<Float>>): Array<Array<Float>> {
        var idx_end: Int = -1;
        val t0 = dataArray[0][0];

        for (k in 0 until dataArray[0].size - 1) {

            // Clip data at a 10 seconds time-frame
            if (dataArray[0][k] - t0 >= 1000.0) {
                idx_end = k;
                break;
            // OR use all samples if the dataset contains less than 10 seconds.
            } else if (dataArray[0][k] < 0) {
                idx_end = k-1
                break;
            }
        }

        if (idx_end == -1) {
            idx_end = dataArray[0].size - 1
        }

        return arrayOf(
                dataArray[0].sliceArray(0..idx_end),
                dataArray[1].sliceArray(0..idx_end),
                dataArray[2].sliceArray(0..idx_end),
                dataArray[3].sliceArray(0..idx_end)
        )
    }

    private fun resampleTimeSeries(dataArray: Array<Array<Float>>): Array<Array<Float>> {
        val t_old = dataArray[0]
        val t0 = t_old[0]
        val t_new = t_old.map { tk -> tk - t0 };

        val N_samples: Int = (t_new[t_new.size - 1]* sampling_frequency).toInt();
        val x1 = Array<Float>(size=N_samples, init={0.0F});
        val x2 = Array<Float>(size=N_samples, init={0.0F});
        val x3 = Array<Float>(size=N_samples, init={0.0F});

        var i: Int = 0;

        for (k: Int in 0 until N_samples - 1) {

            val tk = k / sampling_frequency;

            if (tk > t_new[i + 1]) {
                i++;
            }

            val ti = t_new[i];
            val tip1 = t_new[i+1];

            val x1_i = dataArray[1][i];
            val x1_ip1 = dataArray[1][i+1];

            val x2_i = dataArray[2][i];
            val x2_ip1 = dataArray[2][i+1];

            val x3_i = dataArray[3][i];
            val x3_ip1 = dataArray[3][i+1];

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


    private fun removeOffsets(gyroSensorData: Array<Array<Float>>,
                              accelSensorData: Array<Array<Float>>): Array<Array<Array<Float>>> {

        val gyroDataWithoutOffset = removeOffset(gyroSensorData)
        val accelDataWithoutOffset = removeOffset(accelSensorData)

        return arrayOf(gyroDataWithoutOffset, accelDataWithoutOffset)
    }

    private fun removeOffset(dataArray: Array<Array<Float>>): Array<Array<Float>> {

        val offset1 = dataArray[0].average();
        val offset2 = dataArray[1].average();
        val offset3 = dataArray[2].average();

        return arrayOf(
                dataArray[0].map { x -> (x - offset1).toFloat() }.toTypedArray(),
                dataArray[1].map { x -> (x - offset2).toFloat() }.toTypedArray(),
                dataArray[2].map { x -> (x - offset3).toFloat() }.toTypedArray()
        )
    }

    private fun calcEnergies(gyroSensorData: Array<Array<Float>>,
                             accelSensorData: Array<Array<Float>>): Array<Array<Float>> {

        val gyroSensorEnergies: Array<Float> = calcEnergy(gyroSensorData);
        val accelSensorEnergies: Array<Float> = calcEnergy(accelSensorData);

        return arrayOf(gyroSensorEnergies, accelSensorEnergies)
    }

    private fun calcEnergy(dataArray: Array<Array<Float>>): Array<Float> {
        val N_samples = dataArray[0].size;

        val x1_squared: Array<Float> = dataArray[0].map { x -> x * x / N_samples }.toTypedArray();
        val x2_squared: Array<Float> = dataArray[1].map { x -> x * x / N_samples}.toTypedArray();
        val x3_squared: Array<Float> = dataArray[2].map { x -> x * x / N_samples}.toTypedArray();

        return arrayOf(
                x1_squared.sum(),
                x2_squared.sum(),
                x3_squared.sum()
        )
    }

    private fun doClassification(gyroEnergies: Array<Float>, accelEnergies: Array<Float>): Int {
        val energy_vector: Array<Float> = arrayOf(
                gyroEnergies[0],
                gyroEnergies[1],
                gyroEnergies[2],
                accelEnergies[0],
                accelEnergies[1],
                accelEnergies[2]
        );

        // Calculate euclidean distances to reference individuals.
        val euclideanDistances = Array<Float>(size=neighbors.size, init={0.0F});
        for (k: Int in 0 until neighbors.size) {
            euclideanDistances[k] = euclideanDistance(energy_vector, neighbors[k]);
        }

        val neighbor_indices: IntArray = (0..neighbors.size - 1).toList().toIntArray();
        val sorted_neighbor_indices = neighbor_indices.sortedBy { idx -> euclideanDistances[idx] }
        val classes_of_nearest_neighbors = neighbor_classes.sliceArray(sorted_neighbor_indices)

        return kMajorityVoting(classes_of_nearest_neighbors, 3)
    }

    private fun kMajorityVoting(sortedClassesOfNearestNeighbors: Array<Int>, k:Int):Int {
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
        return _class
    }

    private fun euclideanDistance(v1: Array<Float>, v2: Array<Float>): Float {
        val diff = Array<Float>(size = v1.size, init={0.0F});

        // Squared differences
        for (k: Int in 0 until v1.size) {
            diff[k] = (v1[k] - v2[k]).pow(2);
        }

        // Square-root of sum
        return diff.sum().pow(x = 0.5F);
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

    fun read(file:String) : Pair<Array<Array<Float>>, Array<Int>> {

        val referenceData = mutableListOf<Array<Float>>();
        val referenceClasses = mutableListOf<Int>();

        val file_ = File(file);
        val reader = file_.bufferedReader()
        var line: String? = reader.readLine();
        while (line != null) {
            val tokens = line.split(";")
            val tmp = arrayOf<Float>(
                    tokens[1].toFloat(), // Gyro x
                    tokens[2].toFloat(), // Gyro y
                    tokens[3].toFloat(), // Gyro z
                    tokens[4].toFloat(), // Accelerometer x
                    tokens[5].toFloat(), // Accelerometer y
                    tokens[6].toFloat(), // Accelerometer z
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

    fun load(testsetDirectory: String): Triple<List<Array<Array<Float>>>, List<Array<Array<Float>>>, List<Int>> {
        /* The test-set consists of CSV-files for each activity, with two files per activity. (One
        for the gyro-sensor and one for the accelerometer.)

        For each activity, the sensor-data files are located each in a subdirectory.

        The properties <activityDirectories> specify the subdirectories and <activityClasses>
        state the class-id for each subdirectory.

         */

        val gyroData: MutableList<Array<Array<Float>>> = mutableListOf();
        val accelData: MutableList<Array<Array<Float>>> = mutableListOf();
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

        return Triple<List<Array<Array<Float>>>, List<Array<Array<Float>>>, List<Int>> (gyroData, accelData, activityClassPerFilePair)
    }


    private fun readCSVSensorDataFile(file:String): Array<Array<Float>> {

        val data: MutableList<Array<Float>> = mutableListOf();

        val reader: BufferedReader = File(file).bufferedReader()
        var line: String? = reader.readLine();
        while (line != null) {
            val tokens = line.split(";")
            try {
                val tmp = arrayOf<Float>(
                        tokens[0].toFloat(), // Time
                        tokens[1].toFloat(), // Data x
                        tokens[2].toFloat(), // Data y
                        tokens[3].toFloat()  // Data z
                )
                data.add(element = tmp);

            } catch (e: NumberFormatException) {}


            line = reader.readLine()
        }
        reader.close()

        val N_samples = data.size
        val dataArray: Array<Array<Float>> = arrayOf(
                Array<Float>(N_samples, init = {0.0F}),
                Array<Float>(N_samples, init = {0.0F}),
                Array<Float>(N_samples, init = {0.0F}),
                Array<Float>(N_samples, init = {0.0F})
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