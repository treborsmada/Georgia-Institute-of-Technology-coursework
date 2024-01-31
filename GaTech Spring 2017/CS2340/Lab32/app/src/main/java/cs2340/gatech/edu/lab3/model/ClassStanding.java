package cs2340.gatech.edu.lab3.model;

/**
 * Created by Robert Adams on 2/7/2017.
 */

public enum ClassStanding {
    FRESHMAN("FR"),
    SOPHOMORE("SO"),
    JUNIOR("JR"),
    SENIOR("SR");

    private String value;

    ClassStanding(String value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return value;
    }
}
