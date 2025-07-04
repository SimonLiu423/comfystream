import argparse
import asyncio
import json
import logging
import os
import sys
import torch
import cv2
import av

# Initialize CUDA before any other imports to prevent core dump.
if torch.cuda.is_available():
    torch.cuda.init()

from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.codecs import h264
from aiortc.rtcrtpsender import RTCRtpSender
from comfystream.pipeline import Pipeline
from twilio.rest import Client
from comfystream.server.utils import patch_loop_datagram, add_prefix_to_app_routes, FPSMeter
from comfystream.server.metrics import MetricsManager, StreamStatsManager
from utils import SkillMatch, ImageChunkBuffer
import time
logger = logging.getLogger(__name__)
logging.getLogger("aiortc.rtcrtpsender").setLevel(logging.WARNING)
logging.getLogger("aiortc.rtcrtpreceiver").setLevel(logging.WARNING)


MAX_BITRATE = 2000000
MIN_BITRATE = 2000000


class VideoStreamTrack(MediaStreamTrack):
    """video stream track that processes video frames using a pipeline.

    Attributes:
        kind (str): The kind of media, which is "video" for this class.
        track (MediaStreamTrack): The underlying media stream track.
        pipeline (Pipeline): The processing pipeline to apply to each video frame.
    """

    kind = "video"

    def __init__(
            self, track: MediaStreamTrack, pipeline: Pipeline, pc: RTCPeerConnection):
        """Initialize the VideoStreamTrack.

        Args:
            track: The underlying media stream track.
            pipeline: The processing pipeline to apply to each video frame.
            pc: The peer connection to create data channel on.
        """
        super().__init__()
        self.track = track
        self.pipeline = pipeline
        self.fps_meter = FPSMeter(
            metrics_manager=app["metrics_manager"], track_id=track.id
        )
        self.running = True
        self.pc = pc
        self.frame_count = 0  # Add frame counter
        self.skill_match = SkillMatch(5)
        # Create a data channel for sending frame metadata
        try:
            self.data_channel = pc.createDataChannel("frame_metadata")
            logger.info(
                f"Created data channel for frame metadata, initial state: {self.data_channel.readyState}")
        except Exception as e:
            logger.error(f"Failed to create data channel: {str(e)}")
            self.data_channel = None

        # Start frame collection immediately, don't wait for data channel
        self.collect_task = asyncio.create_task(self.collect_frames())

        # Add cleanup when track ends
        @track.on("ended")
        async def on_ended():
            logger.info("Source video track ended, stopping collection")
            await cancel_collect_frames(self)

    async def collect_frames(self):
        """Collect video frames from the underlying track and pass them to
        the processing pipeline. Stops when track ends or connection closes.
        """
        try:
            self.frame_count = 0
            while self.running:
                try:
                    frame = await self.track.recv()
                    # Rotate frame 90 degrees by swapping width/height and transforming pixels
                    rotated_frame = av.VideoFrame.from_ndarray(
                        frame.to_ndarray(format="rgb24").transpose(1, 0, 2)[::-1],
                        format="rgb24"
                    )
                    rotated_frame.pts = frame.pts
                    rotated_frame.time_base = frame.time_base
                    await self.pipeline.put_video_frame(rotated_frame)

                    # self.frame_count += 1
                except asyncio.CancelledError:
                    logger.info("Frame collection cancelled")
                    break
                except Exception as e:
                    if "MediaStreamError" in str(type(e)):
                        logger.info("Media stream ended")
                    else:
                        logger.error(f"Error collecting video frames: {str(e)}")
                    self.running = False
                    break

            # Perform cleanup outside the exception handler
            logger.info("Video frame collection stopped")
        except asyncio.CancelledError:
            logger.info("Frame collection task cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in frame collection: {str(e)}")
        finally:
            await self.pipeline.cleanup()

    async def recv(self):
        """Receive a processed video frame from the pipeline, increment the frame
        count for FPS calculation and return the processed frame to the client.
        """
        processed_frame, original_frame = await self.pipeline.get_processed_video_frame()

        if self.data_channel and self.data_channel.readyState == "open":
            opencv_frame = original_frame.to_ndarray(format="bgr24")
            pose_match = self.skill_match.checkMatchId(opencv_frame)
            metadata = {
                "frame_number": self.frame_count,
                "timestamp": time.time(),
                "width": original_frame.width,
                "height": original_frame.height,
                "pose_match": int(pose_match)
            }
            self.data_channel.send(json.dumps(metadata))

        self.frame_count += 1

        # Save processed frame every 100 frames
        # if self.frame_count % 100 == 0:
        #     try:
        #         opencv_frame = processed_frame.to_ndarray(format="bgr24")
        #         filename = f"processed_frame_{self.frame_count}.png"
        #         cv2.imwrite(filename, opencv_frame)
        #     except Exception as e:
        #         logger.error(f"Error saving processed frame: {str(e)}")

        # Increment the frame count to calculate FPS.
        await self.fps_meter.increment_frame_count()

        return processed_frame


class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track: MediaStreamTrack, pipeline):
        super().__init__()
        self.track = track
        self.pipeline = pipeline
        self.running = True
        self.collect_task = asyncio.create_task(self.collect_frames())

        # Add cleanup when track ends
        @track.on("ended")
        async def on_ended():
            logger.info("Source audio track ended, stopping collection")
            await cancel_collect_frames(self)

    async def collect_frames(self):
        """Collect audio frames from the underlying track and pass them to
        the processing pipeline. Stops when track ends or connection closes.
        """
        try:
            while self.running:
                try:
                    frame = await self.track.recv()
                    await self.pipeline.put_audio_frame(frame)
                except asyncio.CancelledError:
                    logger.info("Audio frame collection cancelled")
                    break
                except Exception as e:
                    if "MediaStreamError" in str(type(e)):
                        logger.info("Media stream ended")
                    else:
                        logger.error(f"Error collecting audio frames: {str(e)}")
                    self.running = False
                    break

            # Perform cleanup outside the exception handler
            logger.info("Audio frame collection stopped")
        except asyncio.CancelledError:
            logger.info("Frame collection task cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in audio frame collection: {str(e)}")
        finally:
            await self.pipeline.cleanup()

    async def recv(self):
        return await self.pipeline.get_processed_audio_frame()


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    codecPrefs = [codec for codec in codecs if codec.mimeType == forced_codec]
    transceiver.setCodecPreferences(codecPrefs)


def get_twilio_token():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if account_sid is None or auth_token is None:
        return None

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token


def get_ice_servers():
    ice_servers = []

    token = get_twilio_token()
    if token is not None:
        # Use Twilio TURN servers
        for server in token.ice_servers:
            if server["url"].startswith("turn:"):
                turn = RTCIceServer(
                    urls=[server["urls"]],
                    credential=server["credential"],
                    username=server["username"],
                )
                ice_servers.append(turn)

    return ice_servers


async def offer(request):
    pipeline = request.app["pipeline"]
    pcs = request.app["pcs"]
    # pose_targets = request.app["pose_targets"]
    image_buffers = request.app["image_buffers"]

    params = await request.json()

    await pipeline.set_prompts(params["prompts"])

    offer_params = params["offer"]
    offer = RTCSessionDescription(sdp=offer_params["sdp"], type=offer_params["type"])

    ice_servers = get_ice_servers()
    if len(ice_servers) > 0:
        logger.info(f"Using ICE servers: {ice_servers}")
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=get_ice_servers())
        )
    else:
        logger.info("No ICE servers found, using default configuration")
        pc = RTCPeerConnection()

    logger.info(f"Creating new peer connection {id(pc)}")

    # Add event handlers to track connection state
    @pc.on("icegatheringstatechange")
    def on_icegatheringstatechange():
        logger.info(f"ICE gathering state is: {pc.iceGatheringState}")

    @pc.on("iceconnectionstatechange")
    def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is: {pc.iceConnectionState}")

    @pc.on("signalingstatechange")
    def on_signalingstatechange():
        logger.info(f"Signaling state is: {pc.signalingState}")

    pcs.add(pc)
    # pose_targets[id(pc)] = default_targets
    image_buffer = ImageChunkBuffer()
    image_buffers[id(pc)] = image_buffer

    tracks = {"video": None, "audio": None}

    # Only add video transcever if video is present in the offer
    if "m=video" in offer.sdp:
        # Prefer h264
        transceiver = pc.addTransceiver("video")
        caps = RTCRtpSender.getCapabilities("video")
        prefs = list(filter(lambda x: x.name == "H264", caps.codecs))
        transceiver.setCodecPreferences(prefs)

        # Monkey patch max and min bitrate to ensure constant bitrate
        h264.MAX_BITRATE = MAX_BITRATE
        h264.MIN_BITRATE = MIN_BITRATE

    # Handle control channel from client
    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"Data channel received from client: {channel.label}")
        if channel.label == "frame_metadata":
            logger.info("Frame metadata channel established")

            @channel.on("open")
            def on_open():
                logger.info("Frame metadata channel opened")

            @channel.on("close")
            def on_close():
                logger.info("Frame metadata channel closed")

            @channel.on("error")
            def on_error(error):
                logger.error(f"Frame metadata channel error: {error}")

            @channel.on("message")
            def on_message(message):
                logger.info(f"Frame metadata channel received message: {message}")

        elif channel.label == "control":
            logger.info("Control channel established")

            @channel.on("open")
            def on_open():
                logger.info("Control channel opened")

            @channel.on("close")
            def on_close():
                logger.info("Control channel closed")

            @channel.on("message")
            async def on_message(message):
                logger.info(f"Control channel received message: {message}")
                try:
                    params = json.loads(message)

                    if params.get("type") == "get_nodes":
                        nodes_info = await pipeline.get_nodes_info()
                        response = {"type": "nodes_info", "nodes": nodes_info}
                        channel.send(json.dumps(response))
                    elif params.get("type") == "update_prompts":
                        if "prompts" not in params:
                            logger.warning(
                                "[Control] Missing prompt in update_prompt message"
                            )
                            return
                        await pipeline.update_prompts(params["prompts"])
                        response = {"type": "prompts_updated", "success": True}
                        channel.send(json.dumps(response))
                    # elif params.get("type") == "set_pose_targets":
                    #     is_complete = image_buffer.add_chunk(params["pose_target_chunk"])
                    #     if is_complete:
                    #         pose_id = params.get("pose_id")
                    #         image = image_buffer.get_complete_image()
                    #         decoded_image = decode_image(image)
                    #         pose_targets[id(pc)][pose_id] = Pose(
                    #             getTargetLandmarks(decoded_image), 5000)

                    #         image_buffer.clear()
                    #         response = {"type": "pose_targets_set", "status": "target_added"}
                    #     else:
                    #         response = {"type": "pose_targets_set", "status": "chunk_added"}
                    #     channel.send(json.dumps(response))
                    else:
                        logger.warning(
                            "[Server] Invalid message format - missing required fields"
                        )
                except json.JSONDecodeError:
                    logger.error("[Server] Invalid JSON received")
                except Exception as e:
                    logger.error(f"[Server] Error processing message: {str(e)}")

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        if track.kind == "video":
            videoTrack = VideoStreamTrack(track, pipeline, pc)
            tracks["video"] = videoTrack
            sender = pc.addTrack(videoTrack)

            # Store video track in app for stats.
            stream_id = track.id
            request.app["video_tracks"][stream_id] = videoTrack

            codec = "video/H264"
            force_codec(pc, sender, codec)
        elif track.kind == "audio":
            audioTrack = AudioStreamTrack(track, pipeline)
            tracks["audio"] = audioTrack
            pc.addTrack(audioTrack)

        @track.on("ended")
        async def on_ended():
            logger.info(f"{track.kind} track ended")
            request.app["video_tracks"].pop(track.id, None)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
        elif pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)

    if "m=audio" in pc.remoteDescription.sdp:
        await pipeline.warm_audio()
    if "m=video" in pc.remoteDescription.sdp:
        await pipeline.warm_video()

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def cancel_collect_frames(track):
    track.running = False
    if hasattr(track, 'collect_task') is not None and not track.collect_task.done():
        try:
            track.collect_task.cancel()
            await track.collect_task
        except (asyncio.CancelledError):
            pass


async def set_prompt(request):
    pipeline = request.app["pipeline"]

    prompt = await request.json()
    await pipeline.set_prompts(prompt)

    return web.Response(content_type="application/json", text="OK")


def health(_):
    return web.Response(content_type="application/json", text="OK")


async def on_startup(app: web.Application):
    if app["media_ports"]:
        patch_loop_datagram(app["media_ports"])

    app["image_buffers"] = dict()
    # app["pose_targets"] = dict()
    app["pipeline"] = Pipeline(
        cwd=app["workspace"], disable_cuda_malloc=True, gpu_only=True, preview_method='none'
    )
    app["pcs"] = set()
    app["video_tracks"] = {}


async def on_shutdown(app: web.Application):
    pcs = app["pcs"]
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comfystream server")
    parser.add_argument("--port", default=8889, help="Set the signaling port")
    parser.add_argument(
        "--media-ports", default=None, help="Set the UDP ports for WebRTC media"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Set the host")
    parser.add_argument(
        "--workspace", default=None, required=True, help="Set Comfy workspace"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--monitor",
        default=False,
        action="store_true",
        help="Start a Prometheus metrics endpoint for monitoring.",
    )
    parser.add_argument(
        "--stream-id-label",
        default=False,
        action="store_true",
        help="Include stream ID as a label in Prometheus metrics.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    app = web.Application()
    app["media_ports"] = args.media_ports.split(",") if args.media_ports else None
    app["workspace"] = args.workspace

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/", health)
    app.router.add_get("/health", health)

    # WebRTC signalling and control routes.
    app.router.add_post("/offer", offer)
    app.router.add_post("/prompt", set_prompt)

    # Add routes for getting stream statistics.
    stream_stats_manager = StreamStatsManager(app)
    app.router.add_get(
        "/streams/stats", stream_stats_manager.collect_all_stream_metrics
    )
    app.router.add_get(
        "/stream/{stream_id}/stats", stream_stats_manager.collect_stream_metrics_by_id
    )

    # Add Prometheus metrics endpoint.
    app["metrics_manager"] = MetricsManager(include_stream_id=args.stream_id_label)
    if args.monitor:
        app["metrics_manager"].enable()
        logger.info(
            f"Monitoring enabled - Prometheus metrics available at: "
            f"http://{args.host}:{args.port}/metrics"
        )
        app.router.add_get("/metrics", app["metrics_manager"].metrics_handler)

    # Add hosted platform route prefix.
    # NOTE: This ensures that the local and hosted experiences have consistent routes.
    add_prefix_to_app_routes(app, "/live")

    def force_print(*args, **kwargs):
        print(*args, **kwargs, flush=True)
        sys.stdout.flush()

    web.run_app(app, host=args.host, port=int(args.port), print=force_print)
