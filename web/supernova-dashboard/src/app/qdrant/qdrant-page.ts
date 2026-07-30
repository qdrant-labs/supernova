import { Component, inject, signal, ChangeDetectionStrategy } from '@angular/core';
import { Qdrant } from './qdrant';

@Component({
  selector: 'app-qdrant-page',
  templateUrl: './qdrant.html',
  changeDetection: ChangeDetectionStrategy.Eager,
  styleUrl: './qdrant.css',
})
export class QdrantPage {
  private readonly qdrant = inject(Qdrant);

  protected readonly collections = signal<string[]>([]);
  protected readonly error = signal('');

  async loadCollections(): Promise<void> {
    try {
      this.collections.set(await this.qdrant.getCollections());
      this.error.set('');
    } catch {
      this.error.set('Failed to load collections from Qdrant via backend API.');
    }
  }
}
