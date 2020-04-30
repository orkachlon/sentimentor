//package libraries.sentimentAnalysis.src.sentimentAnalysis;

import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


/**
 * SentimentAnalyzer
 */
public class SentimentAnalyzer {

    private static SentimentAnalyzer instance = null;

    private static float neutralBuffer = 0.07f;

    public String sentiment = "";

    public String review = "";

    public Float score = 0f;

    private int currBucket = 0;

    private FeatureManager featureManager = new FeatureManager();

    private List<Float> buckets = new ArrayList<>();

    private SentimentAnalyzer() {}

    public static SentimentAnalyzer getInstance() {
        if (SentimentAnalyzer.instance == null) {
            SentimentAnalyzer.instance = new SentimentAnalyzer();
        }
        return SentimentAnalyzer.instance;
    }
    
    public boolean analyzed() {
      return this.featureManager.size() > 0;
    }

    public void analyze(String review) {
        this.review = review;
        this.featureManager.clear();
        this.buckets.clear();
        try {
          getReviewAnalysis();
          System.out.println(this.featureManager.toString());
        } catch (IOException e) {
          System.err.println("IOException occurred when trying to analyze review");
        }
        if (this.featureManager.size() == 0) {
            throw new RuntimeException("No features found!");
        }
    }

    public String adjust(Float score, String review) throws RuntimeException {
        if (this.review.length() == 0) {
            throw new RuntimeException("Must call analyze before adjust!");
        }
        if (this.score - neutralBuffer < score && score < this.score + neutralBuffer) {
          this.featureManager.reset();
          this.currBucket = this.buckets.size() / 2;
          return this.review;
        }
        String synonym, prev, newReview = review;
        Feature lastChanged;
        int bucket = Collections.binarySearch(this.buckets, score);
        if (bucket < 0) {
            bucket = (-1 * bucket) - 1; // the insertion point
        }
        if (bucket < this.currBucket) {
            // for each of these partitions a word needs to be swapped
            // a feature that couldn't be changed shouldn't be iterated over again in the next i
            // the partitions are defined by the amount of features so there should always be
            // exactly enough
            for (int i = this.currBucket; i > bucket; --i) {
                if (this.featureManager.negify()) {
                    try {
                        lastChanged = this.featureManager.getLastChanged();
                    } catch (NoSuchFieldException e) {
                        break;
                    }
                    prev = lastChanged.getPrev();
                    synonym = lastChanged.toString();
                    newReview = newReview.replaceAll(prev, synonym);
                }
            }
        } else if (bucket > this.currBucket) {
            for (int i = this.currBucket; i < bucket; ++i) {
                if (this.featureManager.posify()) {
                    try {
                        lastChanged = this.featureManager.getLastChanged();
                    } catch (NoSuchFieldException e) {
                        break;
                    }
                    prev = lastChanged.getPrev();
                    synonym = lastChanged.toString();
                    newReview = newReview.replaceAll(prev, synonym);
                }
            }
        }
        this.currBucket = bucket;
        return newReview;
    }

    private void getReviewAnalysis() throws IOException {
        // Process build and execution
        //System.out.println(this.review);
        ProcessBuilder builder = new ProcessBuilder("cmd.exe", "/c",
                "cd \"C:\\Users\\Or Kachlon\\Documents\\ml-as-tool\\interactive-ml-project-1\\app\" && " +
                        ".\\SentimentAnalyzer.bat \"" + this.review + "\"");
        builder.redirectErrorStream(true);
        Process process = builder.start();
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        // Strings, Patterns and Matchers
        String line, features, synonyms;
        //noinspection RegExpRedundantEscape
        Pattern classPattern = Pattern
                .compile("^\\['(?<class>POSITIVE|NEGATIVE)',\\s*(?<confidence>[0-9]\\.[0-9]+)\\]$");
        //noinspection RegExpRedundantEscape
        Pattern featuresPattern = Pattern.compile("\\[((?:\\[(?:(?:'\\w+'(?:, )?){3})\\](?:, )?)+)\\]");
        //noinspection RegExpRedundantEscape
        Pattern featurePattern = Pattern.compile("\\[(?<synonyms>(?:'\\w+'(?:, )?){3})\\]");
        Matcher featureMatcher, classMatcher, featuresMatcher;
        while (true) {
            line = reader.readLine();
            if (line == null) {
                break;
            }
            classMatcher = classPattern.matcher(line);
            featuresMatcher = featuresPattern.matcher(line);
            // Match the classification
            if (classMatcher.matches()) {
                this.score = Float.valueOf(classMatcher.group("confidence"));
                this.sentiment = classMatcher.group("class");
            }
            // Match the feature extraction
            if (featuresMatcher.matches()) {
                // The entire features list and their synonyms, matched to be parsed
                features = featuresMatcher.group(1);
                featureMatcher = featurePattern.matcher(features);
                // Parse all features and their synonyms into a Hashtable
                while (featureMatcher.find()) {
                    synonyms = featureMatcher.group("synonyms");
                    this.featureManager.add(new Feature(Arrays.asList(
                            synonyms.replaceAll("'", "").split(", "))));
                }
            }
        }
        this.createBuckets();
    }

    private void createBuckets() {
        // calculate how many pos/neg partitions fit between old and new scores
        float negPartSize = (this.score - neutralBuffer) / this.featureManager.size();
        float posPartSize = (1 - this.score - neutralBuffer) / this.featureManager.size();
        // fill buckets
        // negative parts
        for (int i = 0; i < this.featureManager.size(); ++i) {
            this.buckets.add((i + 1) * negPartSize);
        }
        // neutral buffer
        this.buckets.add(this.score + neutralBuffer);
        // positive parts
        for (int i = 0; i < this.featureManager.size(); ++i) {
            this.buckets.add(this.buckets.get(this.buckets.size() - 1) + posPartSize);
        }
        this.currBucket = this.buckets.size() / 2;
        System.out.println(this.buckets.size() + " buckets created:");
        System.out.println(this.buckets);
    }

    @Override
    public String toString() {
        return this.review;
    }
}
