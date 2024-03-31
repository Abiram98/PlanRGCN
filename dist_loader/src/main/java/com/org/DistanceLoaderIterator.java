package com.org;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.time.StopWatch;

public class DistanceLoaderIterator implements Iterator<HashMap<StringPair, double[]>> {
    String queryLog;
    String distanceDist;
    StopWatch watch;
    int i = 0;
    String base;
    String[] files;
    Set<String> idSet;

    public DistanceLoaderIterator(String queryLog1,
            String distanceDist1) {
        String queryLog = queryLog1;
        String distanceDist = distanceDist1;

        QueryLogReader reader = new QueryLogReader(queryLog);
        try {
            reader.read();
        } catch (IOException e) {
            e.printStackTrace();
        }
        idSet = new HashSet<>(reader.queryIDs);
        File f = new File(distanceDist);
        base = f.getPath();
        files = f.list();
        watch = new StopWatch();
        watch.start();
    }

    @Override
    public boolean hasNext() {
        if (i < files.length) {
            return true;
        }
        return false;
    }

    @Override
    public HashMap<StringPair, double[]> next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        DistanceLoader distLoader = new DistanceLoader();
        distLoader.loadFile(base + "/" + files[i], idSet);
        i++;
        System.out.println("File " + i + ": " + ((double) watch.getTime(TimeUnit.MILLISECONDS)) / 1000 + "\n");
        return distLoader.map;
    }

}
