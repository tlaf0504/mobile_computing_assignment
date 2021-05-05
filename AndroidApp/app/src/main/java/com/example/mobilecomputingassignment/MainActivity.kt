package com.example.mobilecomputingassignment

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

// Libraries for File IO

class MainActivity : AppCompatActivity(),
                     View.OnClickListener {

    private lateinit var bDataCapturing: Button;
    private lateinit var bActivityRecognition: Button;


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Get Buttons and register Click-event listeners
        this.bActivityRecognition = findViewById(R.id.button_activity_recognition);
        this.bDataCapturing = findViewById(R.id.button_data_capturing);

        this.bActivityRecognition.setOnClickListener(this);
        this.bDataCapturing.setOnClickListener(this);

        // Debug-code. Move directly to activity-recognition.
        //val intent = Intent(this, ActivityRecognition::class.java)
        //startActivity(intent);

    }

    override fun onClick(v: View?) {
        if (v?.id == this.bDataCapturing.id) {
            val intent = Intent(this, DataCapturing::class.java)
            startActivity(intent);
        }

        else if (v?.id == this.bActivityRecognition.id) {
            val intent = Intent(this, ActivityRecognition::class.java)
            startActivity(intent);
        }

    }
}