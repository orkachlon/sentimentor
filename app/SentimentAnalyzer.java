import javafx.util.Pair;
import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;

import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * SentimentAnalyzer
 */
public class SentimentAnalyzer {

    private Float base_score;

//    public Float score;

    public String sentiment;

    public String review;

    private List<Feature> features;

    public void analyze(String review) throws IOException {
        this.review = review;
        this.features = new ArrayList<>();
        getReviewAnalysis();
    }

    public String adjust(Float score) throws ValueException {
        if (this.review.length() == 0) {
            throw new ValueException("\"Must call analyze before adjust!\"");
        }
        String synonym, prev, newReview = this.review;
        // TODO make more efficient - don't calculate from original review each time
        float negPartSize = this.base_score / this.features.size();
        float posPartSize = (1 - this.base_score) / this.features.size();
        int part, f;
        Random picker = new Random();
        if (score < this.base_score) {
            // calculate how many partitions fit between old and new scores
            part = (int) Math.ceil((this.base_score - score) / negPartSize);
            // for each of these partitions a word needs to be swapped
            for (int i = 0; i < part; ++i) {
                // look for a random word that can be swapped - guaranteed to find one since partitions are
                // defined by the features and synonyms
                do {
                    f = picker.nextInt(this.features.size());
                    // save feature to swap
                    prev = this.features.get(f).toString();
                } while(!this.features.get(f).negatify());
                // swap feature
                synonym = this.features.get(f).toString();
                newReview = newReview.replaceAll(prev, synonym);
            }
        } else if (score > this.base_score) {
            // calculate how many partitions fit between old and new scores
            part = (int) Math.ceil((score - this.base_score) / posPartSize);
            // for each of these partitions a word needs to be swapped
            for (int i = 0; i < part; ++i) {
                // look for a random word that can be swapped - guaranteed to find one since partitions are
                // defined by the features and synonyms
                do {
                    f = picker.nextInt(this.features.size());
                    // save feature to swap
                    prev = this.features.get(f).toString();
                } while(!this.features.get(f).positify());
                // swap feature
                synonym = this.features.get(f).toString();
                newReview = newReview.replaceAll(prev, synonym);
            }
        }
        // Suggestion: Split (0, 1) into 2 parts - (0, score)U(score, 1)
        // each part will be split into features.size() parts
        // each time the score moves between parts change 1 feature
//        this.score = score;
        return newReview;
    }

    private void getReviewAnalysis() throws IOException {
        // Process build and execution
        ProcessBuilder builder = new ProcessBuilder("cmd.exe", "/c",
                "cd \"C:\\Users\\Or Kachlon\\Documents\\ml-as-tool\\test\" &",
                "SentimentAnalyzer",
                '"' + this.review + '"');
        builder.redirectErrorStream(true);
        Process p = builder.start();
        BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()));
        // Strings, Patterns and Matchers
        String line, features, synonyms;
        Pattern classPattern = Pattern
                .compile("^\\['(?<class>POSITIVE|NEGATIVE)',\\s*(?<confidence>[0-9]\\.[0-9]+)\\]$");
        Pattern featuresPattern = Pattern.compile("^\\[(\\[(?:(?:'\\w+'(?:, )?)+)+\\])\\]$");
        Pattern featurePattern = Pattern.compile("\\[(?<synonyms>(?:'\\w+'(?:, )?)+)\\]");
        Matcher featureMatcher, classMatcher, featuresMatcher;
        while (true) {
            line = r.readLine();
            if (line == null) {
                break;
            }
            classMatcher = classPattern.matcher(line);
            featuresMatcher = featuresPattern.matcher(line);
            // Match the classification
            if (classMatcher.matches()) {
                this.base_score = Float.valueOf(classMatcher.group("confidence"));
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
                    this.features.add(new Feature(Arrays.asList(
                            synonyms.replaceAll("'", "").split(", "))));
                }
            }
        }
    }

    @Override
    public String toString() {
        return this.review;
    }
}