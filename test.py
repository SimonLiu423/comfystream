import asyncio
import json
import logging
import cv2
import random
import base64
import numpy as np
from aiohttp import ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer
import av

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
av.logging.set_level(av.logging.ERROR)


def encode_image(image):
    """Encode OpenCV image for transfer via data channel"""
    success, encoded_img = cv2.imencode('.jpg', image)
    if not success:
        return None

    return base64.b64encode(encoded_img.tobytes()).decode('utf-8')


# Global variables for video display
local_frame = None
remote_frame = None
data_channel = None
control_channel = None
pose_targets = [
    encode_image(cv2.imread("a.jpg")),
    encode_image(cv2.imread("b.jpg")),
    encode_image(cv2.imread("c.jpg")),
]


def convert_frame_to_opencv(frame):
    # Convert to RGB first
    frame = frame.reformat(format="rgb24")
    # Convert to numpy array
    numpy_frame = frame.to_ndarray()
    # Convert RGB to BGR for OpenCV
    opencv_frame = cv2.cvtColor(numpy_frame, cv2.COLOR_RGB2BGR)
    return opencv_frame


def process_frame(frame):
    """Convert video frame to OpenCV format."""
    # logger.info("Processing frame")
    try:
        img = convert_frame_to_opencv(frame)
        if data_channel and data_channel.readyState == "open":
            # logger.info("Sending message to data channel")
            # data_channel.send(json.dumps({"test": "received"}))
            pass

        if control_channel and control_channel.readyState == "open":
            # logger.info("Sending message to control channel")
            # control_channel.send(
            #     json.dumps(
            #         {"type": "set_pose_targets",
            #          "pose_targets": pose_targets}
            #     )
            # )
            pass

            # logger.info("Frame converted to ndarray")
        return img
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return None


async def display_video():
    """Display both video streams in separate windows."""
    logger.info("Starting video display")
    while True:
        if local_frame is not None:
            cv2.imshow('Local Stream', local_frame)
            # logger.info("Displaying local frame")
        if remote_frame is not None:
            cv2.imshow('Remote Stream', remote_frame)
            # logger.info("Displaying remote frame")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        await asyncio.sleep(0.01)


class VideoStreamTrack(MediaStreamTrack):
    """A video stream track that captures from a camera."""

    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0
        self.frame_interval = 30  # Process 1 frame every 30 frames (1fps at 30fps capture)

    async def recv(self):
        global local_frame
        while True:
            frame = await self.track.recv()
            local_frame = process_frame(frame)
            return frame
            self.frame_count += 1

            # Only process every Nth frame
            if self.frame_count % self.frame_interval == 0:
                # Process and return this frame
                if frame:
                    local_frame = process_frame(frame)
                return frame


async def create_offer(pc, session):
    """Create and send an offer to the server."""
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    prompts = json.load(open("basic.json"))
    offer_json = {
        "endpoint": "http://127.0.0.1:8889",
        "offer": {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        },
        "prompts": [prompts],
    }

    logger.info("Sending offer to server")
    # Send the offer to the server
    async with session.post(
        "http://127.0.0.1:8189/api/offer",  # Adjust the URL as needed
        json=offer_json,
    ) as response:
        if response.status != 200:
            logger.error(f"Failed to send offer: {response.status}")
            return

        answer_data = await response.json()
        answer = RTCSessionDescription(
            sdp=answer_data["sdp"], type=answer_data["type"]
        )
        logger.info("Received answer from server")
        await pc.setRemoteDescription(answer)


async def main():
    global data_channel, control_channel
    # Create ICE server configuration
    ice_servers = [
        RTCIceServer(
            urls=["stun:stun.l.google.com:19302"]
        )
    ]

    # Create a peer connection with ICE configuration
    pc = RTCPeerConnection(
        configuration=RTCConfiguration(iceServers=ice_servers)
    )
    data_channel = pc.createDataChannel("frame_metadata")
    control_channel = pc.createDataChannel("control")

    @control_channel.on("open")
    def on_open():
        logger.info("Control channel opened")
        control_channel.send(
            json.dumps(
                {"type": "set_pose_targets", "pose_targets": pose_targets}
            )
        )
    # Set up media stream (e.g., from a webcam)
    logger.info("Setting up media player")
    player = MediaPlayer("0", format="avfoundation", options={
        "video_size": "640x480",
        "framerate": "30",
    })

    # Add local video track
    if player.video:
        wrapped_track = VideoStreamTrack(player.video)
        pc.addTrack(wrapped_track)
        logger.info("Added local video track")
    else:
        logger.error("No video track available from the camera")
        return

    # Handle incoming tracks
    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        if track.kind == "video":
            async def on_remote_frame():
                global remote_frame
                while True:
                    try:
                        frame = await track.recv()
                        remote_frame = process_frame(frame)
                        # logger.info("Remote frame processed successfully")
                    except Exception as e:
                        logger.error(f"Error processing remote frame: {e}")
                    await asyncio.sleep(0.01)

            asyncio.create_task(on_remote_frame())
        elif track.kind == "audio":
            # Handle audio track
            pass

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"Data channel opened: {channel.label}")

        @channel.on("message")
        async def on_message(message):
            try:
                data = json.loads(message)
                logger.info(f"Received message: {data['frame_number']}, {data['pose_match']}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    # Handle connection state changes

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()

    # Handle ICE connection state changes
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()

    # Handle signaling state changes
    @pc.on("signalingstatechange")
    async def on_signalingstatechange():
        logger.info(f"Signaling state is: {pc.signalingState}")

    # Create a session to send the offer
    async with ClientSession() as session:
        await create_offer(pc, session)

        # Start video display task
        logger.info("Starting video display task")
        display_task = asyncio.create_task(display_video())
        logger.info("Video display task started")

        # Keep the connection alive
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            pass
        finally:
            display_task.cancel()
            cv2.destroyAllWindows()
            await pc.close()


def send_image(channel, image):
    """Send image through data channel"""
    if channel and channel.readyState == "open":
        try:
            encoded = encode_image(image)
            if encoded:
                channel.send(json.dumps({
                    "type": "image",
                    "data": encoded
                }))
                return True
            else:
                logger.error("Failed to encode image")
        except Exception as e:
            logger.error(f"Error sending image: {e}")
    return False


def decode_image(image_data):
    image_data = base64.b64decode(image_data)
    image_data = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image


if __name__ == "__main__":
    asyncio.run(main())
