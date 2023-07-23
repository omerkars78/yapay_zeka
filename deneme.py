matchedData = [{"beforeDirection": "arrowdown", "beforeDistance": 10, "beforeMajor": "1", "beforeMinor": "40", "curMajor": "1", "curMinor": "15", "finishMajor": "1", "finishMinor": "67", "id": 1, "mainRouteId": 1, "nextDirection": "arrowup", "nextDistance": 50, "nextMajor": "1", "nextMinor": "67", "nextText": "50 Metre Düz İlerleyiniz", "startMajor": "1", "startMinor": "15"}, {
    "beforeDirection": "arrowdown", "beforeDistance": 10, "beforeMajor": "1", "beforeMinor": "15", "curMajor": "1", "curMinor": "67", "finishMajor": "1", "finishMinor": "67", "id": 2, "mainRouteId": 1, "nextDirection": "arrowup", "nextDistance": 50, "nextMajor": "1", "nextMinor": "67", "nextText": "50 Metre Düz İlerleyiniz", "startMajor": "1", "startMinor": "15"}]



curMajor = "1"
curMinor = "15"  
nextDistance = 15

for data in matchedData:
    if data["curMajor"] == curMajor and data["curMinor"] == curMinor:
        nextDistance = data["nextDistance"]
    break

print(nextDistance) # output: 50






  const matchedData = [
    {
      beforeDirection: 'arrow-down',
      beforeDistance: 10,
      beforeMajor: '1',
      beforeMinor: '40',
      curMajor: 1,
      curMinor: 67,
      finishMajor: 1,
      finishMinor: 15,
      id: 1,
      mainRouteId: 1,
      nextDirection: 'arrow-up',
      nextDistance: 50,
      nextMajor: 1,
      nextMinor: 15,
      nextText: '50 Metre Düz İlerleyiniz',
      startMajor: 1,
      startMinor: 18,
    },
    {
      beforeDirection: 'arrow-down',
      beforeDistance: 10,
      beforeMajor: 1,
      beforeMinor: 67,
      curMajor: 1,
      curMinor: 15,
      finishMajor: 1,
      finishMinor: 15,
      id: 2,
      mainRouteId: 1,
      nextDirection: 'arrow-down',
      nextDistance: 50,
      nextMajor: 1,
      nextMinor: 67,
      nextText: '1000 Metre Düz İlerleyiniz',
      startMajor: 1,
      startMinor: 67,
    },
  ];