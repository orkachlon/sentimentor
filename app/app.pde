
import controlP5.*;

static final String DEFAULT_TEXT = "Write a review!";
static final int MAX_LEN = 400;

String review;
float sentiment;
boolean analyzed;
SentimentAnalyzer analyzer;

// GUI
color BGColor = color(0, 45, 90);
color textBGColor = color(0, 0, 26);
color textColor = color(0, 116, 217);
ControlP5 cp5;
Textarea textBox;
Slider sentimentSlider;
Button submitButton;


void setup() {
    size(720, 480);
    noStroke();
    review = DEFAULT_TEXT;
    
    // Initiate cp5 library
    cp5 = new ControlP5(this);
    // Set up review textBox
    textBox = cp5.addTextarea("txt")
                  .setPosition(10, 10)
                  .setSize(width - 100, height - 100)
                  .setFont(createFont("consolas", 24))
                  .setLineHeight(32)
                  .setColor(textColor)
                  .setColorBackground(textBGColor)
                  //.setColorForeground(color(0, 128, 255))
                  ;
    textBox.setText(review);
    
    // Set up sentiment slider
    sentimentSlider = cp5.addSlider("sentiment")
                      .setPosition(width - 60, 50)
                      .setSize(30, height - 140)
                      .setRange(0.0, 1.0)
                      .setValue(0.5)
                      .setLabelVisible(false)
                      .lock()
                      ;
    // Set up submit button
    submitButton = cp5.addButton("submit")
                   .setPosition(10, height - 80)
                   .setSize(width - 100, 50)
                   ;
    submitButton.getCaptionLabel().setFont(new ControlFont(createFont("consolas", 24)));
    
    // initialize the analyzer
    analyzer = SentimentAnalyzer.getInstance();
    analyzed = false;
}

void draw() {
    background(BGColor);
}

void keyPressed() {
    if (key == RETURN || key == ENTER) {
      submit();
      return;
    }
    analyzed = false;  // This is here because the function can return before the ifs are done
    sentimentSlider.lock();
    if (key == DELETE) {
        review = DEFAULT_TEXT;
    }
    else if (key == BACKSPACE && review.length() > 0 && !review.equals(DEFAULT_TEXT))
    {
        review = review.substring(0, review.length() - 1);
    }
    else if (isText(key) && review.equals(DEFAULT_TEXT))
    {
        review = Character.toString(key);
    }
    else if (isText(key)) {
      if (review.length() >= MAX_LEN) {
        return;
      }
        review = textBox.getText() + Character.toString(key);
    }
    textBox.setText(review);
}

public void submit() {
  if (analyzed) {
    return;
  }
  textBox.setColorBackground(color(69, 69, 69)); // Doesn't work??
  try {
    analyzer.analyze(review);
  } catch (RuntimeException e) {
    println(e);
    textBox.setColorBackground(textBGColor);
    return;
  }
  analyzed = true;
  println(analyzer.sentiment);
  sentiment = analyzer.score;
  println(sentiment);
  sentimentSlider.setValue(sentiment);
  sentimentSlider.unlock();
  textBox.setColorBackground(textBGColor);
}

public void sentiment(float score) {
  if (!analyzed) return;
  review = analyzer.adjust(score, review);
  textBox.setText(review);
}

boolean isText(char c) {
    return (32 <= c && c <= 126);
}
