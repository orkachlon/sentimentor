public class SATester {

    public static void main(String[] args) {
        SentimentAnalyzer sa = SentimentAnalyzer.getInstance("C:\\Users\\Or Kachlon\\Documents\\ml-as-tool\\interactive-ml-project-1\\app");
        String testString = "The movie was awesome!";
        sa.analyze(testString);
        System.out.println(sa.sentiment);
    }
}
