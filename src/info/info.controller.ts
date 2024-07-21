import { Controller, Get, Query } from '@nestjs/common';
import { InfoService } from './info.service';

@Controller('/get_project_info')
export class InfoController {
  constructor(private readonly infoService: InfoService) {}
  
  @Get()
  getProjectInfo(
    @Query('repoUrl') repoUrl: string,
    @Query('question') question: string,
  ): Promise<Object[]> {
    return this.infoService.getProjectInfo(repoUrl, question);
  }
}
