//package libraries.sentimentAnalysis.src.sentimentAnalysis;

import java.util.List;

public class Feature {

    private List<String> synonyms;

    private Integer index;

    private int prev;

    public Feature(List<String> synonyms) {
        this.synonyms = synonyms;
        this.index = synonyms.size() / 2;
        this.prev = this.index;
    }

    public Boolean posify() {
        if (this.index < synonyms.size() - 1) {
            this.prev = this.index;
            this.index++;
//            System.out.println(this.synonyms.get(index - 1) + " -> " + this.synonyms.get(index));
            return true;
        }
        return false;
    }

    public Boolean negify() {
        if (this.index > 0) {
            this.prev = this.index;
            this.index--;
//            System.out.println(this.synonyms.get(index + 1) + " -> " + this.synonyms.get(index));
            return true;
        }
        return false;
    }
    
    public void reset() {
      this.index = this.synonyms.size() / 2;
    }

    public String getPrev() {
        return this.synonyms.get(this.prev);
    }

    @Override
    public String toString() {
        return this.synonyms.get(this.index);
    }

    public int size() {
        return this.synonyms.size();
    }
}
