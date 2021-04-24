package com.example.mobilecomputingassignment

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import kotlinx.coroutines.*
import java.util.*
import java.lang.Thread
import android.util.Log
import java.util.concurrent.locks.*
import java.util.concurrent.*

class ActivityRecognition : AppCompatActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager;
    private lateinit var gyroSensor: Sensor;
    private lateinit var accelSensor: Sensor;
    private lateinit var classificationThread: ClassificationThread;
    private lateinit var classificationTimer: Timer;
    private lateinit var classificationTimerTask: ClassificationTimerTask;

    /* ========== Sensor-data arrays ========== */
    val gyroSensorArray1 = arrayOf(
            Array<Float>(size=1_000, init={-1.0F}),
            Array<Float>(size=1_000, init={0.0F}),
            Array<Float>(size=1_000, init={0.0F}),
            Array<Float>(size=1_000, init={0.0F})
    );

    val accelSensorArray1 = arrayOf(
            Array<Float>(size=1_000, init={-1.0F}),
            Array<Float>(size=1_000, init={0.0F}),
            Array<Float>(size=1_000, init={0.0F}),
            Array<Float>(size=1_000, init={0.0F})
    );

    val gyroSensorArray2 = arrayOf(
            Array<Float>(size=1_000, init={-1.0F}),
            Array<Float>(size=1_000, init={0.0F}),
            Array<Float>(size=1_000, init={0.0F}),
            Array<Float>(size=1_000, init={0.0F})
    );

    val accelSensorArray2 = arrayOf(
            Array<Float>(size=1_000, init={-1.0F}),
            Array<Float>(size=1_000, init={0.0F}),
            Array<Float>(size=1_000, init={0.0F}),
            Array<Float>(size=1_000, init={0.0F})
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


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_recognition)

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
}


class ClassificationThread
    constructor(val activityThread: ActivityRecognition) : Thread() {

    val synchronizationLock = ReentrantLock()
    val synchronizationCondition = synchronizationLock.newCondition()

    // Set this to true to terminate the thread.
    var terminateThread: Boolean = false;

    override fun run() {
        while(!terminateThread) {
            // Wait until the timer-task triggers the classification
            synchronizationLock.lock()
            synchronizationCondition.await()

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

    private fun doComputation(gyroSensorData: Array<Array<Float>>,
                              accelSensorData: Array<Array<Float>>): Int {

        val resampledSensorData = doResampling(gyroSensorData, accelSensorData)
        val sensorDataWithouOffset = removeOffsets(resampledSensorData[0], resampledSensorData[1])
        val energies = calcEnergies(sensorDataWithouOffset[0], sensorDataWithouOffset[1])
        return doClassification(energies[0], energies[1])
    }

    private fun doResampling(gyroSensorData: Array<Array<Float>>,
                             accelSensorData: Array<Array<Float>>): Array<Array<Array<Float>>> {

        val gyroSensorDataResampled = resampleTimeSeries(gyroSensorData)
        val accelSensorDataResampled = resampleTimeSeries(accelSensorData)

        return arrayOf(gyroSensorDataResampled, accelSensorDataResampled)
    }

    private fun resampleTimeSeries(dataArray: Array<Array<Float>>): Array<Array<Float>> {
        val res = arrayOf(
                Array<Float>(size=1_000, init={0.0F}),
                Array<Float>(size=1_000, init={0.0F}),
                Array<Float>(size=1_000, init={0.0F})
        )
        return res
    }


    private fun removeOffsets(gyroSensorData: Array<Array<Float>>,
                              accelSensorData: Array<Array<Float>>): Array<Array<Array<Float>>> {

        val gyroDataWithoutOffset = removeOffset(gyroSensorData)
        val accelDataWithoutOffset = removeOffset(accelSensorData)

        return arrayOf(gyroDataWithoutOffset, accelDataWithoutOffset)
    }

    private fun removeOffset(dataArray: Array<Array<Float>>): Array<Array<Float>> {
        val res = arrayOf(
                Array<Float>(size=1_000, init={0.0F}),
                Array<Float>(size=1_000, init={0.0F}),
                Array<Float>(size=1_000, init={0.0F})
        )
        return res
    }

    private fun calcEnergies(gyroSensorData: Array<Array<Float>>,
                             accelSensorData: Array<Array<Float>>): Array<Array<Float>> {

        val gyroSensorEnergies = arrayOf(
                calcEnergy(gyroSensorData[0]),
                calcEnergy(gyroSensorData[1]),
                calcEnergy(gyroSensorData[2])
        )

        val accelSensorEnergies = arrayOf(
                calcEnergy(accelSensorData[0]),
                calcEnergy(accelSensorData[1]),
                calcEnergy(accelSensorData[2])
        )

        return arrayOf(gyroSensorEnergies, accelSensorEnergies)
    }

    private fun calcEnergy(dataArray: Array<Float>): Float {
        return 0.0F
    }

    private fun doClassification(gyroEnergies: Array<Float>, accelEnergies: Array<Float>): Int {
        return -1
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