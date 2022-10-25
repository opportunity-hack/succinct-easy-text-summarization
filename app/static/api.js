var clearIntervalId;

const userAction = async () => {
  const location = window.location.hostname;
  console.log("Starting...");
  const request_hash = $("#request_hash").attr("value");
  console.log(request_hash);
  if( request_hash == undefined )
  {
    clearInterval(clearIntervalId);
  }

  if (request_hash != undefined){
    let formData = new FormData();
    formData.append('request_hash', request_hash);

    const response = await fetch('/similar/', {
      method: 'POST',
      body: formData // string or object
    });
    const myJson = await response.json(); //extract JSON from the http response
    // do something with myJson
    console.log("dalsjdaslkdjalsdj");
    console.log(myJson);
    console.log("=============^^^");
    if (myJson != null)
    {
      $("#request_hash").prepend("<h3>Results are ready! Refresh this page to see them.</h3>")
      $("#request_spinner").hide();
      console.log("Stopping automated refresh.");
      console.log(clearIntervalId);
      clearInterval(clearIntervalId);
    }
  }
}

console.log("test1");
userAction();
clearIntervalId = setInterval(userAction, 15000);
console.log("test2");
