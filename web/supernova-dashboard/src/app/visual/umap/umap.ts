import {
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  PLATFORM_ID,
  ViewChild,
  inject,
  signal,
  ChangeDetectionStrategy
} from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';

import { Qdrant, UmapPoint } from '../../qdrant/qdrant';
import { runUmap } from './umap-runner';

interface PlotPoint extends UmapPoint {
  x: number;
  y: number;
}

@Component({
  selector: 'app-umap',
  imports: [FormsModule],
  templateUrl: './umap.html',
  changeDetection: ChangeDetectionStrategy.Eager,
  styleUrl: './umap.css',
})
export class Umap implements OnInit, OnDestroy {
  private readonly qdrant = inject(Qdrant);
  private readonly platformId = inject(PLATFORM_ID);

  @ViewChild('chartContainer', { static: true })
  private chartContainer!: ElementRef<HTMLDivElement>;

  protected readonly collections = signal<string[]>([]);
  protected readonly selectedCollection = signal('');
  protected readonly pointLimit = signal(2000);
  protected readonly nNeighbors = signal(15);
  protected readonly minDist = signal(0.1);
  protected readonly status = signal('Select a collection and compute UMAP.');
  protected readonly error = signal('');
  protected readonly busy = signal(false);
  protected readonly progress = signal(0);
  protected readonly pointCount = signal(0);

  private resizeObserver?: ResizeObserver;
  private plotPoints: PlotPoint[] = [];

  ngOnInit(): void {
    if (isPlatformBrowser(this.platformId)) {
      void this.loadCollections();
      if (typeof ResizeObserver !== 'undefined') {
        this.resizeObserver = new ResizeObserver(() => {
          this.renderChart();
        });
        this.resizeObserver.observe(this.chartContainer.nativeElement);
      }
    } else {
      this.status.set('Open in the browser to compute UMAP.');
    }
  }

  ngOnDestroy(): void {
    this.resizeObserver?.disconnect();
  }

  async loadCollections(): Promise<void> {
    try {
      const names = await this.qdrant.getCollections();
      this.collections.set(names);
      if (names.length > 0 && !this.selectedCollection()) {
        this.selectedCollection.set(names[0]);
      }
      this.status.set(names.length > 0 ? 'Ready to compute UMAP.' : 'No collections found.');
    } catch {
      this.error.set('Failed to load collections from Qdrant via backend API.');
    }
  }

  async computeUmap(): Promise<void> {
    const collectionName = this.selectedCollection().trim();
    if (!collectionName || this.busy()) {
      return;
    }

    const limit = this.pointLimit();
    if (limit > 5000) {
      this.error.set('Point limit above 5000 may be slow in the browser.');
    } else {
      this.error.set('');
    }

    this.busy.set(true);
    this.progress.set(0);
    this.status.set('Fetching vectors from Qdrant...');

    try {
      const points = await this.qdrant.scrollVectors(collectionName, limit);
      if (points.length === 0) {
        throw new Error('No vectors found in the selected collection.');
      }

      this.pointCount.set(points.length);
      this.status.set(`Running UMAP on ${points.length} points...`);

      const coordinates = await runUmap(points.map((point) => point.vector), {
        nNeighbors: this.nNeighbors(),
        minDist: this.minDist(),
        onProgress: (epoch, totalEpochs) => {
          this.progress.set(totalEpochs > 0 ? Math.round((epoch / totalEpochs) * 100) : 0);
        },
      });

      this.plotPoints = points.map((point, index) => ({
        ...point,
        ...coordinates[index],
      }));

      this.status.set(`UMAP complete for ${points.length} points.`);
      this.renderChart();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'UMAP computation failed.';
      this.error.set(message);
      this.status.set('UMAP failed.');
    } finally {
      this.busy.set(false);
      this.progress.set(0);
    }
  }

  private renderChart(): void {
    const container = this.chartContainer.nativeElement;
    const svgElement = container.querySelector('svg');
    if (!svgElement || this.plotPoints.length === 0) {
      return;
    }

    const margin = 40;
    const width = Math.max(container.clientWidth - margin * 2, 200);
    const height = Math.max(container.clientHeight - margin * 2, 200);

    d3.select(svgElement).selectAll('*').remove();
    d3.select(svgElement).attr('width', width + margin * 2).attr('height', height + margin * 2);

    const svg = d3
      .select(svgElement)
      .append('g')
      .attr('transform', `translate(${margin},${margin})`);

    const xScale = d3
      .scaleLinear()
      .domain(d3.extent(this.plotPoints, (point) => point.x) as [number, number])
      .nice()
      .range([0, width]);

    const yScale = d3
      .scaleLinear()
      .domain(d3.extent(this.plotPoints, (point) => point.y) as [number, number])
      .nice()
      .range([height, 0]);

    svg.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(xScale));
    svg.append('g').call(d3.axisLeft(yScale));

    const tooltip = d3
      .select(container)
      .selectAll<HTMLDivElement, PlotPoint>('.umap-tooltip')
      .data([null])
      .join('div')
      .attr('class', 'umap-tooltip')
      .style('opacity', 0);

    svg
      .selectAll('circle')
      .data(this.plotPoints)
      .enter()
      .append('circle')
      .attr('cx', (point) => xScale(point.x))
      .attr('cy', (point) => yScale(point.y))
      .attr('r', 4)
      .attr('fill', '#d04a35')
      .attr('opacity', 0.75)
      .on('mouseenter', (event, point) => {
        tooltip
          .style('opacity', 1)
          .html(this.formatTooltip(point))
          .style('left', `${event.offsetX + 12}px`)
          .style('top', `${event.offsetY + 12}px`);
      })
      .on('mousemove', (event) => {
        tooltip.style('left', `${event.offsetX + 12}px`).style('top', `${event.offsetY + 12}px`);
      })
      .on('mouseleave', () => {
        tooltip.style('opacity', 0);
      });
  }

  private formatTooltip(point: PlotPoint): string {
    const payloadPreview = point.payload
      ? Object.entries(point.payload)
          .slice(0, 3)
          .map(([key, value]) => `${key}: ${String(value)}`)
          .join('<br>')
      : 'No payload';

    return `<strong>id:</strong> ${point.id}<br>${payloadPreview}`;
  }
}
