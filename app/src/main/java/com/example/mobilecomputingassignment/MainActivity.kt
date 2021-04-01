package com.example.mobilecomputingassignment

import android.Manifest.permission.READ_EXTERNAL_STORAGE
import android.Manifest.permission.WRITE_EXTERNAL_STORAGE
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothSocket
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.*
import java.util.*

// Libraries for File IO

class MainActivity : AppCompatActivity(), SensorEventListener {


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

    private var lApplicationStartTime: Long = Calendar.getInstance().timeInMillis;

    // Sensor data. First column: timestamps, second column: sensor-values
    // Currently disables as Bluetooth-transmission of sensor-data is currently not required.
    /*
    private val arrSensorData = arrayOf(
            floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
    );
    */


    // ========== Bluetooth stuff
    private lateinit var bluetoothManager: BluetoothManager
    private lateinit var bluetoothAdapter: BluetoothAdapter;
    private lateinit var connectedBluetoothDevice: BluetoothDevice;
    private lateinit var bluetoothSocket: BluetoothSocket;

    private val REQUEST_ENABLE_BT: Int = 1;
    private val bluetoothUUID = UUID.fromString("53eef5bf-6702-4ab6-bcf3-ae02a401d70e");

    private lateinit var inputStream: InputStream;
    private lateinit var outputStream: OutputStream;

    // ========== FileIO Stuff
    private lateinit var fGyroSensorDataFile: FileWriter;
    private lateinit var fAccelerometerSensorDataFile: FileWriter;
    private var bWriteSensorData: Boolean = false;

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


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


        /* ========== Establish bluetooth connection ========== */
        // Currently, sensor-data transmission via Bluetooth is disabled.
        /*
        bluetoothManager = getSystemService(BLUETOOTH_SERVICE) as BluetoothManager;
        bluetoothAdapter = bluetoothManager.adapter

        // Enable bluetooth adapter if necessary
        if (bluetoothAdapter.isEnabled) {
            val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
            startActivityForResult(enableBtIntent, REQUEST_ENABLE_BT)
        }

        // This is ugly. Improve it...
        val connectedBluetoothDevices: MutableSet<BluetoothDevice> =
                bluetoothAdapter.bondedDevices;
        connectedBluetoothDevice = connectedBluetoothDevices.elementAt(0);

        bluetoothSocket = connectedBluetoothDevice.createRfcommSocketToServiceRecord(bluetoothUUID);
        bluetoothSocket.connect();

        // Finally, get the streams for sending and receiving data
        inputStream = bluetoothSocket.inputStream;
        outputStream = bluetoothSocket.outputStream;
         */

        // ========== Create Sensor-Data File
        createSensorDataFiles()

    }

    private fun createSensorDataFiles() {
        val strDate: String = Calendar.getInstance().time.toString()
        val gyroSensorDataFilePath = File(filesDir, "gyro_sensor_data_%s.txt".format(strDate));
        try {
            fGyroSensorDataFile = FileWriter(gyroSensorDataFilePath);
        } catch (e: FileNotFoundException) {
            e.printStackTrace();
        } catch (e: IOException) {
            e.printStackTrace();
        }

        val accelerometerSensorDataFilePath = File(filesDir, "accel_sensor_data_%s.txt".format(strDate));
        try {
            fAccelerometerSensorDataFile = FileWriter(accelerometerSensorDataFilePath);
        } catch (e: FileNotFoundException) {
            e.printStackTrace();
        } catch (e: IOException) {
            e.printStackTrace();
        }

        fGyroSensorDataFile.write("time;x;y;z\n");
        fAccelerometerSensorDataFile.write("time;x;y;z\n");
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

        fGyroSensorDataFile.flush()
        fAccelerometerSensorDataFile.flush()
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

            twGyroX.text = resources.getString(R.string.gyroscope_label_x, fGyroX);
            twGyroY.text = resources.getString(R.string.gyroscope_label_y, fGyroY);
            twGyroZ.text = resources.getString(R.string.gyroscope_label_z, fGyroZ);

            if (bWriteSensorData) {
                val fTimestamp: Float = (Calendar.getInstance().timeInMillis - lApplicationStartTime).toFloat();
                fGyroSensorDataFile.write("%f;%f;%f;%f\n".format(fTimestamp, fGyroX, fGyroY, fGyroZ));
            }
        }

        if (iSensorType == Sensor.TYPE_ACCELEROMETER) {
            val fAccelX : Float = event.values[0];
            val fAccelY : Float = event.values[1];
            val fAccelZ : Float = event.values[2];

            twAccelX.text = resources.getString(R.string.accelerometer_label_x, fAccelX);
            twAccelY.text = resources.getString(R.string.accelerometer_label_y, fAccelY);
            twAccelZ.text = resources.getString(R.string.accelerometer_label_z, fAccelZ);

            if (bWriteSensorData) {
                val fTimestamp: Float = (Calendar.getInstance().timeInMillis - lApplicationStartTime).toFloat();
                fAccelerometerSensorDataFile.write("%f;%f;%f;%f\n".format(fTimestamp, fAccelX, fAccelY, fAccelZ));
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
    }
}