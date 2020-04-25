import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

import controlP5.*; 

import java.util.HashMap; 
import java.util.ArrayList; 
import java.io.File; 
import java.io.BufferedReader; 
import java.io.PrintWriter; 
import java.io.InputStream; 
import java.io.OutputStream; 
import java.io.IOException; 

public class app extends PApplet {



static final String DEFAULT_TEXT = "Write a review!";
static final int MAX_LEN = 286;

String review;
int sentiment;

ControlP5 cp5;
Textarea textBox;
Slider sentimentSlider;


public void setup() {
    
    noStroke();
    review = DEFAULT_TEXT;
    
    // Initiate cp5 library
    cp5 = new ControlP5(this);
    // Set up review textBox
    textBox = cp5.addTextarea("txt")
                  .setPosition(10, 10)
                  .setSize(width - 100, height - 100)
                  .setFont(createFont("arial", 32))
                  .setLineHeight(25)
                  .setColor(color(128, 10, 210))
                  .setColorBackground(color(60, 255,100))
                  .setColorForeground(color(255, 100))
                  ;
    textBox.setText(review);
    
    // Set up sentiment slider
    cp5.addSlider("sentiment")
       .setPosition(width - 60, 50)
       .setSize(30, 300)
       .setRange(0, 100)
       .setValue(50)
       .setLabelVisible(false)
       ;
    // Set up submit button
    cp5.addButton("Submit")
     .setValue(0)
     .setPosition(10, height - 80)
     .setSize(width - 100, 50)
     ;
}

public void draw() {
    background(100, 240, 69);
    //text(review, 10, 10, width - 100, height - 10);
}

public void keyPressed() {
    if (key == DELETE) {
        review = DEFAULT_TEXT;
    } else if (key == RETURN || key == ENTER) {
        // TODO submit review and predict rating
    } else if (key == BACKSPACE && review.length() > 0 && !review.equals(DEFAULT_TEXT)) {
        review = review.substring(0, review.length() - 1);
    } else if (isText(key) && review.equals(DEFAULT_TEXT)) {
        review = Character.toString(key);
    } else if (isText(key)) {
      if (review.length() >= MAX_LEN) {
        return;
      }
        review = textBox.getText() + Character.toString(key);
    }
    textBox.setText(review);
}

public boolean isText(char c) {
    return (32 <= c && c <= 126);
}
  public void settings() {  size(720, 480); }
  static public void main(String[] passedArgs) {
    String[] appletArgs = new String[] { "app" };
    if (passedArgs != null) {
      PApplet.main(concat(appletArgs, passedArgs));
    } else {
      PApplet.main(appletArgs);
    }
  }
}
