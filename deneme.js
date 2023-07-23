const matchedData = [{"beforeDirection": "arrowdown", "beforeDistance": 10, "beforeMajor": "1", "beforeMinor": "40", "curMajor": "1", "curMinor": "15", "finishMajor": "1", "finishMinor": "67", "id": 1, "mainRouteId": 1, "nextDirection": "arrowup", "nextDistance": 50, "nextMajor": "1", "nextMinor": "67", "nextText": "50 Metre Düz İlerleyiniz", "startMajor": "1", "startMinor": "15"}, {"beforeDirection": "arrowdown", "beforeDistance": 10, "beforeMajor": "1", "beforeMinor": "15", "curMajor": "1", "curMinor": "67", "finishMajor": "1", "finishMinor": "67", "id": 2, "mainRouteId": 1, "nextDirection": "arrowdown", "nextDistance": 50, "nextMajor": "1", "nextMinor": "67", "nextText": "1000 Metre Düz İlerleyiniz", "startMajor": "1", "startMinor": "15"}]
const minRssiDevice = {
    rssi: 0,
    major: 1,
    minor: 15,
  }
  const routing = function() {
    for (let i = 0; i < matchedData.length; i++) {
    const datas = matchedData[i];
    if (datas.curMajor === minRssiDevice.major && datas.curMinor === minRssiDevice.minor) {
    console.log(datas.nextText);
    console.log(datas.nextDirection);
    }
    else if (datas.finishMajor === minRssiDevice.major && datas.finishMinor === minRssiDevice.minor) {
    console.log('Hedefe Ulaştınız');
    }
    else {
        console.log(datas.curMajor);
      }
    }}
  
  routing();
  
// for (let i = 0; i < matchedData.length; i++) {
//     const datas = matchedData[i]; 
//     console.log(datas.beforeDirection);
// }
// const routing = async () => {
//     for (let i = 0; i < matchedData.length; i++) {
//       const datas = matchedData[i];
//       for (let j = 0; j < minRssiDevice.length; j++) {
//         const device = minRssiDevice[j];
//         if (
//           datas.curMajor === device.major &&
//           datas.curMinor === device.minor
//         ) {
//           console.log(datas.nextText);
//           console.log(datas.nextDirection);
          
//         }
//         if (
//           datas.finishMajor === device.major &&
//           datas.finishMinor === device.minor
//         ) {
            
//         } else if (
//           device.minor !== datas.curMinor &&
//           device.major !== datas.curMajor
          
//         ) {
//           console.log("Lütfen İleri İlerleyiniz");
//         }
//       }
//     }
//   };

// const matchedData = [{"beforeDirection": "arrowdown", "beforeDistance": 10, "beforeMajor": "1", "beforeMinor": "40", "curMajor": "1", "curMinor": "15", "finishMajor": "1", "finishMinor": "67", "id": 1, "mainRouteId": 1, "nextDirection": "arrowup", "nextDistance": 50, "nextMajor": "1", "nextMinor": "67", "nextText": "50 Metre Düz İlerleyiniz", "startMajor": "1", "startMinor": "15"}, {"beforeDirection": "arrowdown", "beforeDistance": 10, "beforeMajor": "1", "beforeMinor": "15", "curMajor": "1", "curMinor": "67", "finishMajor": "1", "finishMinor": "67", "id": 2, "mainRouteId": 1, "nextDirection": "arrowdown", "nextDistance": 50, "nextMajor": "1", "nextMinor": "67", "nextText": "1000 Metre Düz İlerleyiniz", "startMajor": "1", "startMinor": "15"}]
// const minRssiDevice = {
//   rssi: 0,
//   major: 1,
//   minor: 15,
// }

// const routing = function() {
//   const datas = matchedData.find(data => data.curMajor === minRssiDevice.major && data.curMinor === minRssiDevice.minor);
//   if (datas) {
//     console.log(datas.nextText);
//     console.log(datas.nextDirection);
//   } else {
//    console.log("none değer")
//   }
// };

// routing();



// const matchedData = [{"beforeDirection": "arrowdown", "beforeDistance": 10, "beforeMajor": "1", "beforeMinor": "40", "curMajor": "1", "curMinor": "15", "finishMajor": "1", "finishMinor": "67", "id": 1, "mainRouteId": 1, "nextDirection": "arrowup", "nextDistance": 50, "nextMajor": "1", "nextMinor": "67", "nextText": "50 Metre Düz İlerleyiniz", "startMajor": "1", "startMinor": "15"}, {"beforeDirection": "arrowdown", "beforeDistance": 10, "beforeMajor": "1", "beforeMinor": "15", "curMajor": "1", "curMinor": "67", "finishMajor": "1", "finishMinor": "67", "id": 2, "mainRouteId": 1, "nextDirection": "arrowdown", "nextDistance": 50, "nextMajor": "1", "nextMinor": "67", "nextText": "1000 Metre Düz İlerleyiniz", "startMajor": "1", "startMinor": "15"}]
// const minRssiDevice = {
//   rssi: 0,
//   major: 1,
//   minor: 15,
// }

// const routing = function() {
//   const datas = matchedData.find(data => data.curMajor === minRssiDevice.major && data.curMinor === minRssiDevice.minor);
//   if (datas) {
//     console.log(datas.nextText);
//     console.log(datas.nextDirection);
//   } else {
//     const finalData = matchedData.find(data => data.finishMajor === minRssiDevice.major && data.finishMinor === minRssiDevice.minor);
//     if (finalData) {
//       console.log('Hedefe Ulaştınız');
//     } else {
//       console.log(matchedData[0].curMajor);
//     }
//   }
// };

// routing();

// const age = 18;
// const my_array = [
//   {
//     "name": "Ahmet",
//     "age": 18

//   },
//   {
//     "name": "Mehmet",
//     "age": 20
//   },
//   {
//     "name": "Ali",
//     "age": 25
//   }
// ]

// function findNameWithAge(my_array, age) {
//   for (let i = 0; i < my_array.length; i++) {
//     if (my_array[i].age === age) {
//       console.log(my_array[i].name);
//     }
//   }
// }

// findNameWithAge(my_array, age);