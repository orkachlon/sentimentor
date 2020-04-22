static final String DEFAULT_TEXT = "Write a review!";

String review;



void setup() {
    size(720, 480);
    review = DEFAULT_TEXT;
}

void draw() {
    background(100, 240, 69);
    textSize(32);
    text(review, 10, 10, width - 100, height - 10);
}

void keyTyped() {
    if (key == DELETE) {
        review = DEFAULT_TEXT;
    } else if (key == RETURN || key == ENTER) {
        // TODO submit review and predict rating
    } else if (key == BACKSPACE && review.length() > 0 && !review.equals(DEFAULT_TEXT)) {
        review = review.substring(0, review.length() - 1);
    } else if (isText(key) && review.equals(DEFAULT_TEXT)) {
        review = Character.toString(key);
    } else if (isText(key)){
        review += Character.toString(key);
    }
}

boolean isText(char c) {
    return (32 <= c && c <= 126);
}