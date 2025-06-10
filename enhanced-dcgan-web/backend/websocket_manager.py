# Updated websocket_manager.py with enhanced debugging
import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = {
            "connected_at": datetime.now().isoformat(),
            "id": str(uuid.uuid4())[:8]
        }

        print(f"🔌 WebSocket connected [{self.connection_info[websocket]['id']}]. Total connections: {len(self.active_connections)}")

        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "message": "Connected to Enhanced DCGAN Backend",
            "timestamp": datetime.now().isoformat()
        }, websocket)

    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        if websocket in self.active_connections:
            connection_id = self.connection_info.get(websocket, {}).get("id", "unknown")
            self.active_connections.remove(websocket)
            if websocket in self.connection_info:
                del self.connection_info[websocket]
            print(f"🔌 WebSocket disconnected [{connection_id}]. Remaining connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific WebSocket client"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            print(f"⚠️  WebSocket: No active connections for broadcast")
            return

        message_str = json.dumps(message)
        disconnected = []

        print(f"📤 WebSocket: Broadcasting to {len(self.active_connections)} clients")
        print(f"📦 Message type: {message.get('type', 'unknown')}")

        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                print(f"❌ Error broadcasting to connection: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

        successful_sends = len(self.active_connections) - len(disconnected)
        print(f"✅ WebSocket: Broadcast successful to {successful_sends} clients")

    async def send_training_update(self, training_id: str, data: dict):
        """Send training progress update with enhanced debugging"""
        print(f"📊 WebSocket: Preparing training update for ID: {training_id}")
        print(f"📈 Update data: Epoch {data.get('current_epoch', '?')}/{data.get('total_epochs', '?')}, "
              f"Progress: {data.get('progress_percentage', 0):.1f}%, "
              f"Status: {data.get('status', 'unknown')}")

        message = {
            "type": "training_status",
            "training_id": training_id,
            "data": data,
            "timestamp": data.get("timestamp", datetime.now().isoformat())
        }

        await self.broadcast(message)
        print(f"✅ WebSocket: Training update sent for ID: {training_id}")

    async def send_log_message(self, log_data: dict):
        """Send log message to frontend with debugging"""
        print(f"📝 WebSocket: Preparing log message - [{log_data.get('level', 'info')}] {log_data.get('message', '')[:50]}...")

        message = {
            "type": "log_message",
            "data": {
                "id": str(uuid.uuid4())[:8],
                "timestamp": log_data.get("timestamp", datetime.now().isoformat()),
                "level": log_data.get("level", "info"),
                "message": log_data.get("message", ""),
                "dataset": log_data.get("dataset", "system"),
                "source": log_data.get("source", "system")
            }
        }

        await self.broadcast(message)
        print(f"✅ WebSocket: Log message sent")

    async def send_system_update(self, system_data: dict):
        """Send system status update"""
        message = {
            "type": "system_update",
            "data": system_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)

    async def send_generation_update(self, generation_id: str, data: dict):
        """Send image generation update"""
        message = {
            "type": "generation_status",
            "generation_id": generation_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)

    def get_connection_info(self) -> List[dict]:
        """Get information about all connections"""
        return [
            {
                "id": info["id"],
                "connected_at": info["connected_at"],
                "uptime_seconds": (datetime.now() - datetime.fromisoformat(info["connected_at"])).total_seconds()
            }
            for info in self.connection_info.values()
        ]

# Global instance
websocket_manager = WebSocketManager()