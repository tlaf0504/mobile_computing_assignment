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
    private lateinit var mProximity: Sensor;

    private lateinit var twGyroX: TextView;
    private lateinit var twGyroY: TextView;
    private lateinit var twGyroZ: TextView;
    private lateinit var twProximity: TextView;

    private var lApplicationStartTime: Long = Calendar.getInstance().timeInMillis;

    // Sensor data. First column: timestamps, second column: sensor-values
    private val arrSensorData = arrayOf(
            floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
    );


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
    private lateinit var fProximitySensorDataFile: FileWriter;

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        mSensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager;
        twGyroX = findViewById(R.id.label_gyro_x);
        twGyroY = findViewById(R.id.label_gyro_y);
        twGyroZ = findViewById(R.id.label_gyro_z);
        twProximity = findViewById(R.id.label_proximity);

        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        val isGyroNull = mGyroscope == null;

        mProximity = mSensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY);
        val isProximityNull = mProximity == null;

        val strNoSensorError: String = resources.getString(R.string.error_no_sensor);

        // Set error messages if corresponding sensor is not available.
        if (isGyroNull) {
            twGyroX.text = strNoSensorError;
            twGyroY.text = strNoSensorError;
            twGyroZ.text = strNoSensorError;
        }

        if (isProximityNull) {
            twProximity.text = strNoSensorError;
        }

        // Return if one of the sensor is not available.
        if (isGyroNull || isProximityNull) {
            return;
        }


        /* ========== Establish bluetooth connection ========== */
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
        val gyroSensorDataFilePath = File(filesDir, "gyro_sensor_data.txt");
        try {
            fGyroSensorDataFile = FileWriter(gyroSensorDataFilePath);
        } catch (e: FileNotFoundException) {
            e.printStackTrace();
        } catch (e: IOException) {
            e.printStackTrace();
        }

        val proximitySensorDataFilePath = File(filesDir, "proxy_sensor_data.txt");
        try {
            fProximitySensorDataFile = FileWriter(proximitySensorDataFilePath);
        } catch (e: FileNotFoundException) {
            e.printStackTrace();
        } catch (e: IOException) {
            e.printStackTrace();
        }

        fGyroSensorDataFile.write("time;x;y;z\n");
        fProximitySensorDataFile.write("time;value\n");
    }

    override fun onStart() {
        super.onStart()
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_NORMAL);
        mSensorManager.registerListener(this, mProximity, SensorManager.SENSOR_DELAY_NORMAL);
    }

    override fun onStop() {
        super.onStop()
        mSensorManager.unregisterListener(this, mGyroscope);
        mSensorManager.unregisterListener(this, mProximity);

        fGyroSensorDataFile.flush()
        fProximitySensorDataFile.flush()
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

            val fTimestamp: Float = (Calendar.getInstance().timeInMillis - lApplicationStartTime).toFloat();
            arrSensorData[0][0] = fTimestamp;
            arrSensorData[0][1] = fTimestamp;
            arrSensorData[0][2] = fTimestamp;

            arrSensorData[1][0] = fGyroX;
            arrSensorData[1][1] = fGyroY;
            arrSensorData[1][2] = fGyroZ;

            fGyroSensorDataFile.write("%f;%f;%f;%f\n".format(fTimestamp, fGyroX, fGyroY, fGyroZ));
        }

        if (iSensorType == Sensor.TYPE_PROXIMITY) {
            twProximity.text = resources.getString(R.string.proximity_label, event.values[0]);

            val fTimestamp: Float = (Calendar.getInstance().timeInMillis - lApplicationStartTime).toFloat();
            val fSensorValue: Float = event.values[0];
            fProximitySensorDataFile.write("%f;%f\n".format(fTimestamp, fSensorValue));
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
    }
}