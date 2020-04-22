import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

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
static final int WIDTH = 720;
static final int HEIGHT = 480;

String review;



public void setup() {
    
    review = DEFAULT_TEXT;
}

public void draw() {
    background(100, 240, 69);
    textSize(32);
    text(review, 10, 10, width - 70, height - 200);
}

public void keyTyped() {
    if (key == DELETE) {
        review = DEFAULT_TEXT;
    } else if (key == BACKSPACE && review.length() > 0 && !review.equals(DEFAULT_TEXT)) {
        review = review.substring(0, review.length() - 1);
    } else if (isText(key) && review.equals(DEFAULT_TEXT)) {
        review = Character.toString(key);
    } else if (isText(key)){
        review += Character.toString(key);
    }
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
