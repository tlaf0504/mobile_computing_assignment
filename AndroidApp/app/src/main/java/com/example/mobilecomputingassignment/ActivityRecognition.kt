package com.example.mobilecomputingassignment

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import kotlinx.coroutines.*
import java.util.*
import java.lang.Thread
import android.util.Log
import java.util.concurrent.locks.*
import java.util.concurrent.*

class ActivityRecognition : AppCompatActivity() {

    private lateinit var sensorDataQueue: BlockingQueue<Array<Array<Array<Float>>>>;
    private lateinit var classificationQueue : BlockingQueue<Int>;


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_recognition)
    }
}

class DataCapturingThread: Thread() {

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


    val buffer1Lock = Semaphore(1);
    val buffer2Lock = Semaphore(1);
    var terminateThread: Boolean = false;
    private val writeDataLock = Semaphore(1);
    private var activeBuffer : Int = 0;

    override fun run() {
         while(!terminateThread) {
             writeDataLock.acquire()
             try {
                 TODO("Capture Sensor-Data")
             } finally {
                 writeDataLock.release()
             }
         }

     }

    fun switchBuffer() {
        writeDataLock.acquire()
        try {
            if (activeBuffer == 0) {
                activeBuffer = 1;
            }
            else {
                activeBuffer = 0;
            }

        } finally {
            writeDataLock.release()
        }
    }

    fun lockBuffer(bufferIndex : Int) {
        if (bufferIndex == 1) {
            writeDataLock.acquire()
            buffer1Lock.acquire()
            writeDataLock.release()
        } else {
            writeDataLock.acquire()
            buffer2Lock.acquire()
            writeDataLock.release()
        }
    }

    fun releaseBuffer(bufferIndex: Int) {
        if (bufferIndex == 1) {
            writeDataLock.acquire()
            buffer1Lock.release()
            writeDataLock.release()
        } else {
            writeDataLock.acquire()
            buffer2Lock.release()
            writeDataLock.release()
        }
    }

    fun getInactiveBuffers() : List<Array<Array<Float>>> {
        // Attention: Not thread-safe. Calling switchBuffer() from another thread
        // might lead to data-inconsistencies!.
        if (activeBuffer == 0) {
            return listOf(gyroSensorArray2, accelSensorArray2)
        } else {
            return listOf(gyroSensorArray1, accelSensorArray1)
        }
    }
}


class ClassificationThread
    constructor(val dataCapturing : DataCapturingThread) : Thread() {

    private val synchronizationLock = ReentrantLock()
    val synchronizationCondition = synchronizationLock.newCondition()

    // Set this to true to terminate the thread.
    var terminateThread: Boolean = false;

    override fun run() {
        while(!terminateThread) {
            synchronizationCondition.await()
            dataCapturing.switchBuffer()
            val sensorData = dataCapturing.getInactiveBuffers()
            val activity = doComputation(sensorData[0], sensorData[1])
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
        // Start the classification
        classificationObject.synchronizationCondition.signal()
    }
}