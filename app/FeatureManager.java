import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;


/**
 * A Manager class for the text features. Makes posify/negify gradual over a set of features
 */
public class FeatureManager implements Iterable<Feature> {

    /** The list of features */
    private List<Feature> features;

    /** The index of the next feature to be adjusted */
    private int currFeature;

    /** index of the last adjusted feature */
    private int lastChanged;

    /**
     * Constructor from a list of features
     * @param features list of features to manage
     */
    public FeatureManager(List<Feature> features) {
        this.features = features;
        this.currFeature = 0;
        this.lastChanged = -1;
    }

    /**
     * Default c'tor
     */
    public FeatureManager() {
        this.features = new ArrayList<>();
        this.currFeature = 0;
        this.lastChanged = -1;
    }

    /**
     * Adds a feature to the end of the manager
     * @param feature to add
     */
    public void add(Feature feature) {
        this.features.add(feature);
    }

    /**
     * Posifies the current feature if further posifying is possible
     * @return true iff operation was successful
     */
    public boolean posify() {
        boolean retVal = false;
        if (this.currFeature < this.features.size()) {
            retVal = this.features.get(this.currFeatureToIndex()).posify();
            this.lastChanged = this.currFeatureToIndex();
            this.currFeature++;
//            assert (retVal);
        }
        return retVal;
    }

    /**
     * Negifies the current feature if further negifying is possible
     * @return true iff operation was successful
     */
    public boolean negify() {
        boolean retVal = false;
        if (-this.features.size() < this.currFeature) {
            this.currFeature--;
            this.lastChanged = this.currFeatureToIndex();
            retVal = this.features.get(this.lastChanged).negify();
//            assert (retVal);
        }
        return retVal;
    }

    /**
     * Clears the manager of all features
     */
    public void clear() {
        this.features.clear();
    }

    /**
     * Resets all of the features of this featureManager
     */
    public void reset() {
        for (Feature feature : this.features) {
            feature.reset();
        }
        this.currFeature = 0;
    }

    /**
     * @return the feature which was last changed
     * @throws NoSuchFieldException if no feature was changed yet
     */
    public Feature getLastChanged() throws NoSuchFieldException {
        if (this.lastChanged < 0) {
            throw new NoSuchFieldException("No change was made yet!");
        }
        return this.features.get(this.lastChanged);
    }

    /**
     * @return the amount of features
     */
    public int size() {
        return this.features.size();
    }

    /**
     * @param i index of feature to return
     * @return the i'th feature. Order is the same as a list's
     */
    public Feature get(int i) {
        return this.features.get(i);
    }

    /**
     * @return string representation of the features
     */
    @Override
    public String toString() {
        return this.features.toString();
    }

    /**
     * @return an iterator over the features
     */
    @Override
    public Iterator<Feature> iterator() {
        return this.features.iterator();
    }

    /**
     * @return a translation of the field currFeature to its index in the list of features
     */
    private int currFeatureToIndex() {
        return this.currFeature >= 0 ?
                this.currFeature :
                this.features.size() + this.currFeature;
    }
}
