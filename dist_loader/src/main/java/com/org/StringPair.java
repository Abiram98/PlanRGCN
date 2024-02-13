package com.org;

public class StringPair {
    public String queryID1;
    public String queryID2;

    public StringPair(String queryID1, String queryID2) {
        this.queryID1 = queryID1;
        this.queryID2 = queryID2;
    }

    @Override
    public int hashCode() {
        return queryID1.hashCode() + queryID2.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (!StringPair.class.isAssignableFrom(obj.getClass())) {
            return false;
        }
        StringPair other = (StringPair) obj;
        return (other.queryID1.equals(queryID1) && other.queryID2.equals(queryID2))
                || (other.queryID1.equals(queryID2) && other.queryID2.equals(queryID1));
    }
}
