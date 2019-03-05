
for event in "guatemalaEarthquake2012" "italyEarthquakes2012" "philipinnesFloods2012" "albertaFloods2013" "australiaBushfire2013" "bostonBombings2013" "manilaFloods2013" "queenslandFloods2013" "typhoonYolanda2013" "joplinTornado2011" "chileEarthquake2014" "typhoonHagupit2014" "nepalEarthquake2015" "flSchoolShooting2018" "parisAttacks2015"
do
java -jar TREC-IS-DownloadDataset.jar --noNewLinesInTweet $event > "${event}.json"
done

for event in "costaRicaEarthquake2012" "fireColorado2012" "floodColorado2013" "typhoonPablo2012" "laAirportShooting2013" "westTexasExplosion2013"
do
java -jar TREC-IS-DownloadDataset.jar --noNewLinesInTweet $event > "${event}.json"
done
