import java.util.List;

public class Feature {

    private List<String> synonyms;

    private Integer index;

    public Feature(List<String> synonyms) {
        this.synonyms = synonyms;
        this.index = 1;
    }

    public Boolean positify() {
        if (index < synonyms.size()) {
            index++;
            return true;
        }
        return false;
    }

    public Boolean negatify() {
        if (index > 0) {
            index--;
            return true;
        }
        return false;
    }

    @Override
    public String toString() {
        return this.synonyms.get(this.index);
    }

    public int size() {
        return this.synonyms.size();
    }
}