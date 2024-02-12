package com.org;

import java.io.FileNotFoundException;
import java.util.Iterator;
import java.util.LinkedList;

public class MainDistance {
    public static void main(String[] args) {
        DistanceLoader distLoader = new DistanceLoader();
        distLoader.loadFile("");
        System.exit(-1);
        String task = args[0];
        switch (task) {
            case "test": {
                break;
            }
            default: {
                break;
            }
        }

    }
}
