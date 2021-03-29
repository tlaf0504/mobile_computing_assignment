package com.example.mobilecomputingassignment

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.TextView
import android.bluetooth.*
import android.content.Intent
import java.util.*

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var mSensorManager: SensorManager;

    private lateinit var mGyroscope: Sensor;
    private lateinit var mProximity: Sensor;

    private lateinit var twGyroX: TextView;
    private lateinit var twGyroY: TextView;
    private lateinit var twGyroZ: TextView;
    private lateinit var twProximity: TextView;

    private lateinit var bluetoothManager: BluetoothManager
    private lateinit var bluetoothAdapter: BluetoothAdapter;
    private lateinit var connectedBluetoothDevice: BluetoothDevice;
    private lateinit var bluetoothSocket: BluetoothSocket;

    private val REQUEST_ENABLE_BT: Int = 1;

    private val bluetoothUUID = UUID.fromString("53eef5bf-6702-4ab6-bcf3-ae02a401d70e");

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


        bluetoothManager = getSystemService(BLUETOOTH_SERVICE) as BluetoothManager;
        bluetoothAdapter = bluetoothManager.adapter


        if (bluetoothAdapter.isEnabled == false) {
            val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
            startActivityForResult(enableBtIntent, REQUEST_ENABLE_BT)
        }

        val connectedBluetoothDevices: MutableSet<BluetoothDevice> =
                bluetoothAdapter.bondedDevices;

        connectedBluetoothDevice = connectedBluetoothDevices.elementAt(0);

        bluetoothSocket = connectedBluetoothDevice.createRfcommSocketToServiceRecord(bluetoothUUID);
        bluetoothSocket.connect();
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
        }

        if (iSensorType == Sensor.TYPE_PROXIMITY) {
            twProximity.text = resources.getString(R.string.proximity_label, event.values[0]);
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
    }
}