import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


/**
 * A class to analyze reviews and have the option to adjust them based on score
 */
public class SentimentAnalyzer {

    /** default conda path to use if none was given */
    private static final String DEFAULT_CONDA_PATH = "C:/ProgramData/Anaconda3/Scripts";

    /** The single instance of this Singleton class*/
    private static SentimentAnalyzer instance = null;

    /** The sentiment: 'POSITIVE'/'NEGATIVE' */
    public String sentiment;

    /** The review */
    public String review;

    /** path to conda command */
    private String condaPath;

    /** path to sketch directory */
    private String sketchPath;

    /** The score given to the review at analysis */
    public Float score;

    /** Current bucket from buckets that the review is in */
    private int currBucket;

    /** The feature manager for the last analyzed review */
    private FeatureManager featureManager;

    /** List of buckets in the range 0-1 to match the adjustment variation levels */
    private List<Float> buckets;

    /**
     * private constructor for Singleton design
     */
    private SentimentAnalyzer(String sketchPath, String condaPath) {
        this.sentiment = "";
        this.review = "";
        this.score = 0f;
        this.currBucket = 0;
        this.featureManager = new FeatureManager();
        this.buckets = new ArrayList<>();
        this.condaPath = condaPath;
        this.sketchPath = sketchPath;
    }

    /**
     * @return the single instance of this class with given conda path
     */
    public static SentimentAnalyzer getInstance(String sketchPath, String condaPath) {
        if (SentimentAnalyzer.instance == null) {
            SentimentAnalyzer.instance = new SentimentAnalyzer(sketchPath, condaPath);
        }
        SentimentAnalyzer.instance.condaPath = condaPath;
        SentimentAnalyzer.instance.sketchPath = sketchPath;
        return SentimentAnalyzer.instance;
    }

    /**
     * @return the single instance of this class with default conda path: C:\ProgramData\Anaconda3\Scripts
     */
    public static SentimentAnalyzer getInstance(String sketchPath) {
        return SentimentAnalyzer.getInstance(sketchPath, DEFAULT_CONDA_PATH);
    }

    /**
     * Analyzes the given review and saves its features for later adjustment
     * @param review the review to analyze
     */
    public void analyze(String review) {
        this.review = review;
        this.featureManager.clear();
        this.buckets.clear();
        try {
          getReviewAnalysis();
        } catch (IOException e) {
          System.err.println("IOException occurred when trying to analyze review");
        }
        if (this.featureManager.size() == 0) {
            throw new RuntimeException("No features found!");
        }
    }

    /**
     * Matches the review sentiment to the given score. Must call analyze beforehand.
     * @param score  the score to make the review match to
     * @return the review adjusted to fit the given score
     * @throws RuntimeException if not review has been analyzed yet
     */
    public String adjust(Float score) throws RuntimeException {
        if (this.review.length() == 0) {
            throw new RuntimeException("Must call analyze before adjust!");
        }
        String synonym, prev;
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
                    this.review = this.review.replaceAll(prev, synonym);
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
                    this.review = this.review.replaceAll(prev, synonym);
                }
            }
        }
        this.currBucket = bucket;
        return this.review;
    }

    /**
     * Uses the python part of the program to analyze the given review
     * @throws IOException if the reader fails to read the process
     */
    private void getReviewAnalysis() throws IOException {
        // Process build and execution
        String nlp_dir = this.sketchPath.substring(0, this.sketchPath.lastIndexOf('\\')) + "\\dev";
        String cmd = "cd \"" + this.sketchPath + "\" && " +
                ".\\SentimentAnalyzer.bat \"" +
                this.condaPath + "\" \"" +
                nlp_dir + "\" \"" +
                this.review + "\"";
        ProcessBuilder builder = new ProcessBuilder("cmd.exe", "/c", cmd);
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

    /**
     * Creates the partitions of 0-1 based on the amount of features found
     */
    private void createBuckets() {
        // calculate how many pos/neg partitions fit between old and new scores
        float neutralBuffer = 0.07f;
        float negPartSize = (this.score - neutralBuffer) / this.featureManager.size();
        float posPartSize = (1 - this.score - neutralBuffer) / this.featureManager.size();
        // fill buckets
        // negative parts
        if (negPartSize > 0) {
            for (int i = 0; i < this.featureManager.size(); ++i) {
                this.buckets.add((i + 1) * negPartSize);
            }
        }
        // neutral buffer
        this.buckets.add(this.score + neutralBuffer);
        // positive parts
        if (posPartSize > 0) {
            for (int i = 0; i < this.featureManager.size() - 1; ++i) {
                this.buckets.add(this.buckets.get(this.buckets.size() - 1) + posPartSize);
            }
        }
        // last bucket is always 1
        this.buckets.add(1f);
        this.currBucket = this.buckets.size() / 2;
    }

    /**
     * @return The review at its current state
     */
    @Override
    public String toString() {
        return this.review;
    }

    /**
     * @return true iff a review has been analyzed
     */
    public boolean analyzed() {
        return this.featureManager.size() > 0;
    }
}
