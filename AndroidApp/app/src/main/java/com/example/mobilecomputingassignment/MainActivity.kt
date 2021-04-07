package com.example.mobilecomputingassignment

import android.Manifest.permission.READ_EXTERNAL_STORAGE
import android.Manifest.permission.WRITE_EXTERNAL_STORAGE
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import java.io.*
import java.util.*
import java.util.concurrent.locks.Lock
import java.util.concurrent.locks.ReentrantLock

// Libraries for File IO

class MainActivity : AppCompatActivity(), SensorEventListener, CompoundButton.OnCheckedChangeListener, AdapterView.OnItemSelectedListener {


    // ========== Sensor-data stuff
    private lateinit var mSensorManager: SensorManager;

    private lateinit var mGyroscope: Sensor;
    private lateinit var mAccelerometer: Sensor;

    private lateinit var twGyroX: TextView;
    private lateinit var twGyroY: TextView;
    private lateinit var twGyroZ: TextView;
    private lateinit var twAccelX: TextView;
    private lateinit var twAccelY: TextView;
    private lateinit var twAccelZ: TextView;
    private lateinit var spActivitySelectionSpinner: Spinner;

    private var captureStartTimestamp: Long = Calendar.getInstance().timeInMillis;


    /* Create 2D-Arrays for storing sensor-data. Each row contains 4 float-values:
    * 1.) The Timestamp
    * 2.) The sensors x-value
    * 3.) The sensors y-value
    * 4.) The sensors z-value. */

    private val gyroSensorArray = arrayOf(
            Array<Float>(size=1_000_000, init={-1.0F}),
            Array<Float>(size=1_000_000, init={0.0F}),
            Array<Float>(size=1_000_000, init={0.0F}),
            Array<Float>(size=1_000_000, init={0.0F})
    );

    private val accelSensorArray = arrayOf(
            Array<Float>(size=1_000_000, init={-1.0F}),
            Array<Float>(size=1_000_000, init={0.0F}),
            Array<Float>(size=1_000_000, init={0.0F}),
            Array<Float>(size=1_000_000, init={0.0F})
    );

    // Counters used by the sensor-listeners to fill the sensor-arrays.
    // Int is used as Kotlin in version 1.3 only supports unsigned integers as experimental feature.
    private var gyroSensorFillCounter : Int = 0;
    private var accelSensorFillCounter : Int = 0;

    // ========== FileIO Stuff
    private lateinit var fGyroSensorDataFile: FileWriter;
    private lateinit var fAccelerometerSensorDataFile: FileWriter;

    private lateinit var strSelectedActivity: String;

    private lateinit var swSensorDataRecordingSwitch: Switch;

    private var bWriteSensorData: Boolean = false;
    private val writeSensorDataLock: Lock = ReentrantLock();


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        /* Initialization-code taken from
         * https://developer.android.com/guide/topics/ui/controls/spinner (2021-04-07)
         */
        spActivitySelectionSpinner = findViewById(R.id.activity_selection_spinner)
        spActivitySelectionSpinner.onItemSelectedListener = this // Register Listener

        ArrayAdapter.createFromResource(
                this,
                R.array.activity_selection_spinner_entries,
                android.R.layout.simple_spinner_item
        ).also {adapter ->
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            spActivitySelectionSpinner.adapter = adapter}

        /* ========== Initialize sensors ========== */
        mSensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager;
        twGyroX = findViewById(R.id.label_gyro_x);
        twGyroY = findViewById(R.id.label_gyro_y);
        twGyroZ = findViewById(R.id.label_gyro_z);
        twAccelX = findViewById(R.id.label_accel_x);
        twAccelY = findViewById(R.id.label_accel_y);
        twAccelZ = findViewById(R.id.label_accel_z);


        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        val isGyroNull = mGyroscope == null;

        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        val isAccelNull = mAccelerometer == null;

        val strNoSensorError: String = resources.getString(R.string.error_no_sensor);


        // Set error messages if corresponding sensor is not available.
        if (isGyroNull) {
            twGyroX.text = strNoSensorError;
            twGyroY.text = strNoSensorError;
            twGyroZ.text = strNoSensorError;
        }

        if (isAccelNull) {
            twAccelX.text = strNoSensorError;
            twAccelY.text = strNoSensorError;
            twAccelZ.text = strNoSensorError;
        }


        // Return if one of the sensor is not available.
        if (isGyroNull) {
            return;
        }

        // Register event-listener for sensor-data recording switch.
        swSensorDataRecordingSwitch = findViewById(R.id.recording_switch);
        swSensorDataRecordingSwitch.setOnCheckedChangeListener(this);
        spActivitySelectionSpinner.selectedItem

    }

    private fun createSensorDataFiles() {
        var strDate: String = Calendar.getInstance().timeInMillis.toString()
        strDate = strDate.replace(oldValue = " ", newValue = "_")

        val gyroSensorDataFilePath = File(filesDir, "%s_gyro_sensor_data_%s.txt".format(strSelectedActivity, strDate));
        try {
            fGyroSensorDataFile = FileWriter(gyroSensorDataFilePath);
        } catch (e: FileNotFoundException) {
            e.printStackTrace();
        } catch (e: IOException) {
            e.printStackTrace();
        }

        val accelerometerSensorDataFilePath = File(filesDir, "%s_accel_sensor_data_%s.txt".format(strSelectedActivity, strDate));
        try {
            fAccelerometerSensorDataFile = FileWriter(accelerometerSensorDataFilePath);
        } catch (e: FileNotFoundException) {
            e.printStackTrace();
        } catch (e: IOException) {
            e.printStackTrace();
        }

        // Write the file-headers. Use a '#' to indicate a comment-line.
        fGyroSensorDataFile.write("#time;x;y;z\n");
        fAccelerometerSensorDataFile.write("#time;x;y;z\n");
    }

    private fun flushAndCloseSensorDataFiles() {
        fAccelerometerSensorDataFile.flush();
        fAccelerometerSensorDataFile.close();

        fGyroSensorDataFile.flush();
        fGyroSensorDataFile.close();

    }

    private fun writeAllSensorData() {
        writeArrayToFile(gyroSensorArray, fGyroSensorDataFile);
        writeArrayToFile(accelSensorArray, fAccelerometerSensorDataFile);
    }

    private fun writeArrayToFile(array: Array<Array<Float>>, file: FileWriter) {
        val N_rows : Int = array[0].size

        for (k : Int in 0 until N_rows) {
            /* The first column states the time-stamp of the measurement. This column is filled
            *  with -1 when starting a new capture. The first value < 0.0
            *  therefore indicated the end of the current caputre.
            */
            if (array[0][k] >= 0.0) {
                val t : Float = array[0][k]
                val x : Float = array[1][k]
                val y : Float = array[2][k]
                val z : Float = array[3][k]

                file.write("%f;%f;%f;%f\n".format(t, x, y, z));
            }
            else { break; }
        }
    }

    /* Clean the content of the given sensor-data array. The first column is filled with (-1)s,
    the remaining 3 columns are filled with zeros.
    As the array is successively filled in the sensor-event listener, one can determine the total
    amount of measurement samples by searching for the first negative value in the first column.
     */
    private fun cleanSensorDataArray(array: Array<Array<Float>>) {
        array[0].fill(element=-1.0F)
        array[1].fill(element=0.0F)
        array[2].fill(element=0.0F)
        array[3].fill(element=0.0F)
    }

    override fun onStart() {
        super.onStart()
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_NORMAL);
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
    }

    override fun onStop() {
        super.onStop()
        mSensorManager.unregisterListener(this, mGyroscope);
        mSensorManager.unregisterListener(this, mAccelerometer);
    }

    override fun onSensorChanged(event: SensorEvent?) {

        if (event == null){
            return;
        }

        val iSensorType: Int = event.sensor.type;

        if (iSensorType == Sensor.TYPE_GYROSCOPE) {
            val fGyroX : Float = event.values[0];
            val fGyroY : Float = event.values[1];
            val fGyroZ : Float = event.values[2];

            /* The code below is uncommented on purpose to show the current sensor values on the
            *  screen. In the release version it stays commented to save computation-time.*/
            /*
            twGyroX.text = resources.getString(R.string.gyroscope_label_x, fGyroX);
            twGyroY.text = resources.getString(R.string.gyroscope_label_y, fGyroY);
            twGyroZ.text = resources.getString(R.string.gyroscope_label_z, fGyroZ);
            */

            writeSensorDataLock.lock()
            try {
                if (bWriteSensorData && gyroSensorFillCounter < gyroSensorArray[0].size) {
                    gyroSensorArray[0][gyroSensorFillCounter] = (Calendar.getInstance().timeInMillis - captureStartTimestamp).toFloat();
                    gyroSensorArray[1][gyroSensorFillCounter] = fGyroX;
                    gyroSensorArray[2][gyroSensorFillCounter] = fGyroY;
                    gyroSensorArray[3][gyroSensorFillCounter] = fGyroZ;

                    gyroSensorFillCounter++;

                }
            } finally {
                writeSensorDataLock.unlock()
            }

        }

        if (iSensorType == Sensor.TYPE_ACCELEROMETER) {
            val fAccelX : Float = event.values[0];
            val fAccelY : Float = event.values[1];
            val fAccelZ : Float = event.values[2];

            /* The code below is uncommented on purpose to show the current sensor values on the
            *  screen. In the release version it stays commented to save computation-time.*/
            /*
            twAccelX.text = resources.getString(R.string.accelerometer_label_x, fAccelX);
            twAccelY.text = resources.getString(R.string.accelerometer_label_y, fAccelY);
            twAccelZ.text = resources.getString(R.string.accelerometer_label_z, fAccelZ);
            */

            writeSensorDataLock.lock()
            try {
                if (bWriteSensorData && accelSensorFillCounter < accelSensorArray[0].size) {
                    accelSensorArray[0][accelSensorFillCounter] = (Calendar.getInstance().timeInMillis - captureStartTimestamp).toFloat();
                    accelSensorArray[1][accelSensorFillCounter] = fAccelX;
                    accelSensorArray[2][accelSensorFillCounter] = fAccelY;
                    accelSensorArray[3][accelSensorFillCounter] = fAccelZ;

                    accelSensorFillCounter++;
                }
            } finally {
                writeSensorDataLock.unlock()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
    }

    /* Listener for data-capturing button state change. */
    override fun onCheckedChanged(buttonView: CompoundButton?, isChecked: Boolean) {
        writeSensorDataLock.lock()
        try {
            if (isChecked) {
                cleanSensorDataArray(gyroSensorArray);
                gyroSensorFillCounter = 0;

                cleanSensorDataArray(accelSensorArray);
                accelSensorFillCounter = 0;

                captureStartTimestamp = Calendar.getInstance().timeInMillis
                bWriteSensorData = true;

                spActivitySelectionSpinner.isEnabled = false;
            }
            else {
                bWriteSensorData = false;
                createSensorDataFiles();
                writeAllSensorData();
                flushAndCloseSensorDataFiles();

                spActivitySelectionSpinner.isEnabled = true;
            }
        } finally {
            writeSensorDataLock.unlock()
        }
    }

    // Callback for Spinner (AdapterView.OnItemSelectedListener)
    override fun onItemSelected(parent: AdapterView<*>, view: View?, pos: Int, id: Long) {

        // An item was selected. You can retrieve the selected item using
        // parent.getItemAtPosition(pos)
        val selected_item = parent.getItemAtPosition(pos)
        strSelectedActivity = selected_item.toString()

    }

    // Must be implemented by class for AdapterView.OnItemSelectedListener
    override fun onNothingSelected(parent: AdapterView<*>) {
    }
}