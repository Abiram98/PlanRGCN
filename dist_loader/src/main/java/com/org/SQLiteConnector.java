package com.org;

import java.io.File;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.time.StopWatch;

public class SQLiteConnector {
    Connection c = null;
    String sqliteDBfile;
    String queryLog;
    String distanceFolder;
    int commitLimit = 5000;
    int limit = 10;

    public SQLiteConnector(String sqliteDBfile1, String qLog, String distF) {
        queryLog = qLog;
        sqliteDBfile = sqliteDBfile1;
        distanceFolder = distF;
    }

    public void runner() {
        try {
            Class.forName("org.sqlite.JDBC");
            c = DriverManager.getConnection("jdbc:sqlite:" + sqliteDBfile);
            File f_db = new File(sqliteDBfile);
            if (!f_db.exists())
                createTable();
            c.setAutoCommit(false);
            DistanceLoaderIterator iter = new DistanceLoaderIterator(queryLog, distanceFolder);
            int count = 0;
            while (iter.hasNext()) {
                HashMap<StringPair, double[]> map = iter.next();
                for (StringPair s : map.keySet()) {
                    double[] dist = map.get(s);
                    if (dist == null) {
                        continue;
                    }
                    if (insertGED(s.queryID1, s.queryID2, dist[0])) {
                        count++;
                    }
                    if (count >= limit) {
                        break;
                    }

                    if (count % commitLimit == 0) {
                    }

                }
                commit();
            }
        } catch (Exception e) {
            System.err.println(e.getClass().getName() + ": " + e.getMessage());
            System.exit(0);
        }
        System.out.println("Opened database successfully");
    }

    public void executeUpdateQuery(String query) throws SQLException {
        if (c == null) {
            System.out.println("Connection is null");
            System.exit(-1);
        }

        Statement stmt = c.createStatement();
        stmt.executeUpdate(query);
        stmt.close();

    }

    public void createTable() throws SQLException {
        String createTable = "CREATE TABLE ged(query_pair varchar(510), queryid1 varchar(255),queryid2 varchar(255),dist real, PRIMARY KEY (query_pair));";
        executeUpdateQuery(createTable);
        c.commit();
    }

    public boolean insertGED(String queryID1, String queryID2, double dist) {
        String query = String.format("""
                INSERT INTO ged VALUES ('%s_%s' ,'%s', '%s', %.3f)
                    """, queryID1, queryID2, queryID1, queryID2, dist);
        try {
            executeUpdateQuery(query);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return false;
        }

        return true;
    }

    public boolean commit() {
        try {
            c.commit();
        } catch (Exception e) {
            System.err.println(e.getClass().getName() + ": " + e.getMessage());
            return false;
        }
        return true;
    }

    public void close() throws SQLException {
        c.close();
    }

}
