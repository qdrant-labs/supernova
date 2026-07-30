import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';

export interface CursorAgentStatus {
  ready: boolean;
  agentId?: string;
  error?: string;
}

export interface CursorChatResponse {
  agentId?: string;
  runId: string;
  status: string;
  text: string;
  error?: string;
}

@Injectable({ providedIn: 'root' })
export class CursorDevService {
  private readonly http = inject(HttpClient);

  getStatus() {
    return this.http.get<CursorAgentStatus>('/api/cursor/status');
  }

  chat(message: string) {
    return this.http.post<CursorChatResponse>('/api/cursor/chat', { message });
  }
}
