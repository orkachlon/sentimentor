public class SATester {

    public static void main(String[] args) {
        SentimentAnalyzer sa = SentimentAnalyzer.getInstance();
        String testString = "Nice movie nice acting";
        sa.analyze(testString);
    }
}
