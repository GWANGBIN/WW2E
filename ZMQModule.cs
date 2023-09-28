using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using System;
using UnityEngine;

// USAGE: 1. Instantiate, 2. Call Start(), 3. Call Stop().
// Stops requesting when isRunning == false.

public class ZMQModule: ZMQThread
{
    int count = 0;


    protected override void Run()
    {
        // Required to prevent Unity from freezing at 2nd run.
        ForceDotNet.Force();

        using(RequestSocket client = new RequestSocket()) {
            client.Connect("tcp://127.0.0.1:5555");

            while(true) {
                ///// SEND /////

                // Send data if update interval is over and fetched Render Texture exists
                if((Global.Variables.deltaTime > Global.Variables.updateInterval)
                    && (Global.Variables.sourceImgCount > 0)) {
                    // Send texure
                    string str = Convert.ToBase64String(Global.Variables.sourceImg);
                    client.SendFrame(str);
                    Debug.Log("[INFO]  #" + ++count + " in thread  |  DeltaTime " + Global.Variables.deltaTime +
                              "  |  FPS " + (1 / Global.Variables.deltaTime) + "  |  AccumTime " + Global.Variables.accumulatedTime);

                    // Reset variables for the next iteration
                    Global.Variables.deltaTime = 0;
                }

                // Send empty frame otherwise: Unity will crash if don't
                else {
                    //Debug.Log("[INFO]  Sending empty frame...");
                    client.SendFrameEmpty();
                }

                // Tell the main thread to fetch new Render Texture
                // *) RenderTexture.active can only be accessed from the main thread
                Global.Variables.sourceFetchTrigger = true;



                ///// RECEIVE /////
                // Listen to incoming data
                string message = null;
                bool received = false;

                while(isRunning) {
                    received = client.TryReceiveFrameString(out message);
                    if(received)
                        break;
                }

                // Add to the queue
                if(received && !(string.Equals(message, ""))) {
                    // Trim b' and '
                    message = message.Trim('b');
                    message = message.Trim('\'');


                    // If image received
                    if(message.Length > 10) {
                        byte[] img = Convert.FromBase64String(message);
                        Global.Variables.resultImg = img;
                        Global.Variables.resultShow = true;
                    }

                    // If not received (probability below threshold)
                    else if (message.Length > 0) {
                    Debug.Log(message.Length);
                        Global.Variables.resultShow = false;
                    }

                }
            }
        }
    }
}
