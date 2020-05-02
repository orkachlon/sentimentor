import java.util.List;


/**
 * A representation of a feature and all of its negative/positive synonyms
 */
public class Feature {

    /** list of synonyms of this feature */
    private List<String> synonyms;

    /** index of the current highlighted synonym */
    private int index;

    /** index of the previously highlighted synonym */
    private int prev;

    /**
     * Constructor from list of synonyms
     * @param synonyms list of synonyms to use, middle synonym is considered the base
     */
    public Feature(List<String> synonyms) {
        this.synonyms = synonyms;
        this.index = synonyms.size() / 2;
        this.prev = this.index;
    }

    /**
     * Attempts to posify this feature
     * @return true iff operation was successful
     */
    public Boolean posify() {
        if (this.index < synonyms.size() - 1) {
            this.prev = this.index;
            this.index++;
//            System.out.println(this.synonyms.get(index - 1) + " -> " + this.synonyms.get(index));
            return true;
        }
        return false;
    }

    /**
     * Attempts to negify this feature
     * @return true iff operation was successful
     */
    public Boolean negify() {
        if (this.index > 0) {
            this.prev = this.index;
            this.index--;
//            System.out.println(this.synonyms.get(index + 1) + " -> " + this.synonyms.get(index));
            return true;
        }
        return false;
    }

    /**
     * resets the highlighting to the feature's base synonym
     */
    public void reset() {
      this.index = this.synonyms.size() / 2;
    }

    /**
     * @return the previously highlighted synonym
     */
    public String getPrev() {
        return this.synonyms.get(this.prev);
    }

    /**
     * @return the current highlighted synonym of this feature
     */
    @Override
    public String toString() {
        return this.synonyms.get(this.index);
    }

    /**
     * @return The amount of synonyms this feature has
     */
    public int size() {
        return this.synonyms.size();
    }
}
