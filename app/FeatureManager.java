import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;


/**
 * A Manager class for the text features
 */
public class FeatureManager implements Iterable<Feature> {

    private List<Feature> features;

    private int currFeature;

    private int lastChanged;

    public FeatureManager(List<Feature> features) {
        this.features = features;
        this.currFeature = 0;
        this.lastChanged = -1;
    }

    public FeatureManager() {
        this.features = new ArrayList<>();
        this.currFeature = 0;
        this.lastChanged = -1;
    }

    public void add(Feature feature) {
        this.features.add(feature);
    }

    public boolean posify() {
        boolean retVal = false;
        if (this.currFeature < this.features.size()) {
            retVal = this.features.get(this.currFeatureToIndex()).posify();
            this.lastChanged = this.currFeatureToIndex();
            this.currFeature++;
            assert (retVal);
        }
        return retVal;
    }

    public boolean negify() {
        boolean retVal = false;
        if (-this.features.size() < this.currFeature) {
            this.currFeature--;
            this.lastChanged = this.currFeatureToIndex();
            retVal = this.features.get(this.lastChanged).negify();
            assert (retVal);
        }
        return retVal;
    }

    public void clear() {
        this.features.clear();
    }

    public void reset() {
        for (Feature feature : this.features) {
            feature.reset();
        }
        this.currFeature = 0;
    }

    public Feature getLastChanged() throws NoSuchFieldException {
        if (this.lastChanged < 0) {
            throw new NoSuchFieldException("No change was made yet!");
        }
        return this.features.get(this.lastChanged);
    }

    public int size() {
        return this.features.size();
    }

    @Override
    public String toString() {
        return this.features.toString();
    }

    @Override
    public Iterator<Feature> iterator() {
        return this.features.iterator();
    }

    private int currFeatureToIndex() {
        return this.currFeature >= 0 ?
                this.currFeature :
                this.features.size() + this.currFeature;
    }
}
